import numpy as np
import mdtraj as md
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


############# Provide mito_shape dimesnions in nanometers ########################  INPUT variables

l_cylinder = 30
r_cylinder = 10
r_junction = 10
l_flat     = 56

#################### Center of the mito membrane ####################################################

center_coords=md.load('../dummy-ref.pdb')                            #### INPUT varibale ####
#print (center_coords.xyz.squeeze())
#np.savetxt('squeeze.txt', center_coords.xyz.squeeze())
#print (center_coords.xyz)
#np.savetxt('no-squeeze.txt', center_coords.xyz) '''save error ValueError: Expected 1D or 2D array, got 3D array instead '''
#np.savetxt('no-squeeze.txt',center_coords.xyz.reshape((3,-1)), fmt="%s", header=str(center_coords.xyz.shape))

        ################# z-com calculation ################################################################

z_com = (center_coords.xyz.squeeze()[:, 2].max() - center_coords.xyz.squeeze()[:, 2].min()) / 2.0
#print (z_com)
#with open('z_com.txt', 'w') as f:
#    f.write(str(z_com))
#f.write(z_com) ----- without str says argument must be a string not numpy float

        ################### xy- COM calculations ############################################################

#print (np.abs(center_coords.xyz.squeeze()[:, 2]))     #prints the z-coordinate of all atoms for pdb/trajectory
xy_inrange = np.abs(center_coords.xyz.squeeze()[:, 2] - z_com) < l_cylinder / 2   #cylinder length dummy atoms inner and outer
#np.savetxt ('xy_inrange.txt',center_coords.xyz.squeeze()[xy_inrange, 0:2])
#print (center_coords.xyz.squeeze()[xy_inrange, 0:2])   ####prints the xy-coordinates which falls under the above equation
xy_means = center_coords.xyz.squeeze()[xy_inrange, 0:2].mean(axis=0)
print (xy_inrange)
print (xy_means)
#np.savetxt ('xy_mean.txt',center_coords.xyz.squeeze()[xy_inrange, 0:2].mean(axis=0))  #COM toph x and y dimesnions - dummy atoms in the cylinder inner and outer regions

        #################### mito membrane center X Y Z #############################################################

mito_center = np.array((xy_means[0], xy_means[1], z_com))
print (mito_center)
#np.savetxt ('mito_center.txt',np.array((xy_means[0], xy_means[1], z_com)))

################ loading trajectory for lipid partition calculation #########################

traj = md.load('../top_hg-4us.xtc',top='../top_hg.pdb', stride=1)                                         ### INPUT variables ######
#print (traj)
print (traj.xyz, traj.unitcell_lengths, mito_center)
#print (traj.xyz.shape)   1250 frames, 13650 atoms, 3 (coordinates)
pdb = md.load('../top_hg.pdb')
POPC = pdb.topology.select("resname POPC")
TEST = pdb.topology.select("resname TEST")
#print ('printing indices')
#print (POPC)

################### calculation of rescaled mito membrane center over the trajectory and vector calculations ################

mito_center_scaled = mito_center[np.newaxis, :].repeat(traj.xyz.shape[1], axis=0)[np.newaxis, :, :]    #copy the center for all atoms in the trajectory
#print (mito_center_scaled.ndim)
#print (mito_center_scaled)
#np.savetxt('mito_center_scaled.txt',mito_center_scaled.reshape((3,-1)), fmt="%s", header=str(mito_center_scaled.shape))
if traj.xyz.shape[0] > 1:
        mito_center_scaled = mito_center_scaled.repeat(traj.xyz.shape[0], axis=0)
        print (mito_center_scaled)  # prtints the same for every frame
        if not mito_center_scaled.ndim == 3:
            raise ValueError("coordinates should be nframes * nparticles * ndims, p_origin shape = {}".format(mito_center_scaled.shape))   # noqa
        if not traj.unitcell_lengths.ndim == 2:
            raise ValueError("boxdims should be nframes * nparticles, boxdims shape = {}".format(traj.unitcell_lengths.shape))
        if not mito_center_scaled.shape == traj.xyz.shape:
            raise ValueError("input vector dimension mismatch. Origin shape = {}, destination shape =  {}".format(
                         mito_center_scaled.shape, traj.xyz.shape))
        if not mito_center_scaled.shape[0] == traj.unitcell_lengths.shape[0]:  # mismatch between number of frames in coords and boxdims
            raise ValueError("Mismatch between number of frames in coordinates ({}) and boxdims ({})".format(
                         mito_center_scaled.shape[0], traj.unitcell_lengths.shape[0]))
        if not mito_center_scaled.shape[2] == traj.unitcell_lengths.shape[1]:  # mismatch between dimensionality
            raise ValueError("Mismatch between number of dimensions in coordinates ({}) and boxdims ({})".format(
                         mito_center_scaled.shape[2], traj.unitcell_lengths.shape[1]))
        boxdims_reshaped = traj.unitcell_lengths[:, np.newaxis, :]  # allows broadcasting
        boxdims_midpoint = boxdims_reshaped / 2
        vecs = traj.xyz - mito_center_scaled
        print ('print vectors')
        print (vecs)
        print ('&&&&&')
   #     print (len(vecs))     ###number of frames
        print (vecs.shape)     ###number of frames 1251, number of atoms 13650, 3 boxdimensions
        #veclengths = np.abs(vecs)
       # np.savetxt('vecl.txt', veclengths.reshape((3,-1)), fmt="%s")

    # these are the vectors who's periodic image are closer than the original vecotor
        #vecs_gt_boxdims = veclengths >  (boxdims_midpoint)  # these positions will be changed

    # boolean arrays for identifying closest periodic image - based on vector direction instead of
    # place in box, which might not be centered on (0, 0, 0)
        #negative_vecs = vecs < 0
        #positive_vecs = vecs > 0

    # for positive vectors greater than half the box, use previous periodic image
        #vecs[vecs_gt_boxdims & positive_vecs] = -(boxdims_reshaped - veclengths)[vecs_gt_boxdims & positive_vecs]

    # for negative vectors greater than half the box, use next periodic image.
        #vecs[vecs_gt_boxdims & negative_vecs] = (boxdims_reshaped - veclengths)[vecs_gt_boxdims & negative_vecs]
        print (vecs.squeeze())
        vecs_sqz = vecs.squeeze()
        
        if len(vecs_sqz.shape) == 2:
            x, y, z = vecs_sqz[:, 0], vecs_sqz[:, 1], vecs_sqz[:, 2]
        elif len(vecs_sqz.shape) == 3:
            x, y, z = vecs_sqz[:, :, 0], vecs_sqz[:, :, 1], vecs_sqz[:, :, 2]
            #print (x, y, z)   #3 coordinates for all atoms and all frames aligned in rows ---- x dimesnions - 21362 atoms and 2501 rows

#np.savetxt('vecs_sqz.txt',vecs_sqz)
################ vector transformation cartesian to polar coordinates ############
    
theta    = np.arctan2(y, x)     #######theta is n_parts, angular coordinates  - range is -pi to pi
rho  = np.sqrt(x**2 + y**2)    ######## rho is n_parts, radial coordinates
#print (theta, rho, z)
#np.savetxt('theta.txt',theta)
#np.savetxt('rho.txt',rho)
#np.savetxt('z.txt',z)

################# mapping to mito membrane and -unified coordinate ############

in_cylinder = np.abs(z) < (l_cylinder / 2)
#np.savetxt('cylinder.txt',in_cylinder)                    ### corresponds to 12703890
in_junction = (np.abs(z) >= (l_cylinder / 2)) & (rho <= (r_cylinder + r_junction))
#np.savetxt('junction.txt',in_junction)                   ### to 13279953
in_flat     = rho > (r_cylinder + r_junction)
#np.savetxt('flat.txt',in_flat)                           #### to 27442519
junc_offset = l_cylinder / 2
flat_offset = junc_offset + np.pi * r_junction / 2
test = pd.DataFrame(in_cylinder)
#print ("shapeeeeeeeeeeeeeeeeeeeeeeee")
#print (test)    ####2501 x 21362                      equals the 53426362 , sumf of the in_cyl, in_jun, in_flat info 
##### map to cyl_regime #############
def map_to_cyl_regime(z, mito_shape):
    ''' The mapping of the cylindrical regime is just 0 = center, l_cylinder  / 2 = edge'''
    return np.abs(z)

############## map to junction regime #########''' Angular description only to start - as a distance based one would require averaging the radii of all particles in the description. 0 degrees  = cylinder end, 90 degrees = flat end'''

centered_z = np.abs(z[in_junction]) - (l_cylinder / 2)
centered_rho = rho[in_junction] - r_cylinder - r_junction
map_to_junction = 180 - np.arctan2(centered_z, centered_rho) * 180 / np.pi
#print (map_to_junction)
#np.savetxt ('map_to_junction.txt', map_to_junction)
#np.savetxt ('map_to_junction-center.txt', centered_z)
#np.savetxt ('map_to_junction-rho.txt', centered_rho)

############# map to flat regime ######### ''' rho of 0 is the edge of the junction - no outer edge '''
map_to_flat = rho[in_flat] - r_cylinder - r_junction
#print (map_to_flat)
#np.savetxt ('map_to_flat.txt', map_to_flat)

#print (z.shape)
unified_coordinate = np.zeros(z.shape)
#print (unified_coordinate.shape)
#print (unified_coordinate)

########## output of the z-coordinate of the atoms that are presented in the cylinder region for all the 2501 frames and all the 21362 ''' 

unified_coordinate[in_cylinder] = np.abs(z[in_cylinder]) 
#np.savetxt ('unified_coordinate_cyl.txt', np.abs(z[in_cylinder]))
unified_coordinate[in_junction] = (map_to_junction) * r_junction / (180 / np.pi) + junc_offset  # noqa
#np.savetxt ('unified_coordinate_junc.txt', np.sort(unified_coordinate[in_junction]))
unified_coordinate[in_flat] = (map_to_flat) + flat_offset
#np.savetxt ('unified_coordinate_flat.txt', np.sort(unified_coordinate[in_flat]))
#print (unified_coordinate.shape)
#np.savetxt ('unified_coordinate.txt', unified_coordinate)



print ('top_cylinder region:' + str (unified_coordinate[in_cylinder].max()))
print ('top_junction region:' + str (unified_coordinate[in_junction].max()))
print ('top_flat region:' + str (unified_coordinate[in_flat].max()))




############################## bins distribution ####################

bins = [0]
points_per_bin = 300
sorted_data  = np.sort(unified_coordinate[0, :])
#np.savetxt ('sorted.txt', sorted_data)    # note that we don't topher getting the max value - when we digitize, anything right
    # of the maximum is just another data point
print (bins)
bins = np.array (bins + [sorted_data[i] for i in range(points_per_bin, sorted_data.size, points_per_bin )]) #split the dorted data based on the points_per_bin i.e., 21362 atoms divides by 1000 if points per bin is 1000 ---- number of bins would bw 22 in this case... every 1000 point is the bin limit
labeling = np.array ([sorted_data[i] for i in range(points_per_bin, sorted_data.size, points_per_bin )]) #split the dorted data based on the points_per_bin i.e., 21362 atoms divides by 1000 if points per bin is 1000 ---- number of bins would bw 22 in this case... every 1000 point is the bin limit
print (bins)
#format_bins =  "{%.1f}" % (bins)
#format_bins =  "{:.1f}".format(str(bins))
format_bins =  list(map("{:.1f}".format, labeling))
print (format_bins)
np.savetxt('bin_label-top.txt', format_bins, fmt='%s')
digitized_coords = np.digitize(unified_coordinate, bins)    #### notify the presence of the atoms in each frame and corresponding bin index
#test1 = pd.DataFrame (digitized_coords)
print (digitized_coords.shape)
#print (test1)
#np.savetxt ('digitized_coordinate.txt', digitized_coords)
digitized_coords[digitized_coords == digitized_coords.max()] = digitized_coords.max() - 1 ############ did not understand

# gather coordinate info on new bin centers
nbins = 1 + digitized_coords.max() - digitized_coords.min()    # nbins = number of bins
min_bin = digitized_coords.min()
print (min_bin)
bin_unified = np.zeros(nbins)
bin_z = np.zeros(nbins)
bin_rho = np.zeros(nbins)

for i in range(nbins):
    bin_unified[i] = unified_coordinate[digitized_coords == (i + min_bin)].mean()
    #print (bin_unified)
    bin_z[i]       = np.abs(z[digitized_coords == (i + min_bin)]).mean()
    bin_rho[i]     = rho[digitized_coords == (i + min_bin)].mean()

#print ('sorted---data;;')
#print ([sorted_data[i] for i in range(points_per_bin, sorted_data.size, points_per_bin )])
print (bin_rho)
print (bin_z)
print (bin_unified)

np.savetxt('bin_rho_top.txt', bin_rho)
np.savetxt('bin_z_top.txt', bin_z)
#np.savetxt('bin_unified.txt', bin_unified)

n_frames = digitized_coords.shape[0]
#print (n_frames)
#print (digitized_coords.max())
#print (digitized_coords.min())
n_bins = 1 + digitized_coords.max() - digitized_coords.min()
#print ('binsssssssssssss and new bins')

#print  (bins)
#print (n_bins)

PC_fract = np.zeros((n_frames, n_bins))
popc_bin = np.zeros((n_frames, n_bins))
for i in range(digitized_coords.min(), digitized_coords.max() + 1):
    digitized_mask = (digitized_coords == i)
    PC_fract[:, i - 1] = digitized_mask[:, POPC].sum(axis=1) / digitized_mask.sum(axis=1)
    
    ##########total leaflet lipid calculations'
    popc_bin[:, i-1]=(digitized_mask[:, POPC].sum(axis=1))
np.savetxt('popc_bin.txt', popc_bin)
total_popc =pd.DataFrame(popc_bin/10297)
total_popc2=pd.DataFrame(popc_bin)
############testing POPC####
print ('total POPC')
print (total_popc)
print (total_popc2)
print (total_popc.sum(axis=1))
print (total_popc2.sum(axis=1))
print ((total_popc2.sum(axis=1))/10297)
#########################################################################################
print (total_popc)
print ('is it two columns')
print (total_popc.iloc[:, 0:2])
#top_popc_cyl = ((total_popc.iloc[:, 0:9]).sum(axis=1))
#print ((total_popc.iloc[:, 0:9]-total_popc.iloc[0,:]))
#print ((total_popc.iloc[:, 0:9]-total_popc.iloc[0,:]).sum(axis=1))
print ('cecking junction')
#print ((total_popc.iloc[:, 9:24]))
#print ((total_popc.iloc[:, 9:24]-total_popc.iloc[0,:]))
#print ((total_popc.iloc[:, 9:24]-total_popc.iloc[0,:]).sum(axis=1))
##########mapping flat
print ((total_popc.iloc[:, 24:]))
print ((total_popc.iloc[:, 24:]-total_popc.iloc[0,:]))
print ((total_popc.iloc[:, 24:]-total_popc.iloc[0,:]).sum(axis=1))
###########################################################################################
################with respect to the concentration
total_popc_cyl_def = pd.DataFrame(((total_popc.iloc[:, 0:8].sum(axis=1))))
#total_popc_cyl_def = pd.DataFrame((10534-(total_popc.iloc[:, 0:8].sum(axis=1))))
print ('checking new cylinder regions')
print (total_popc_cyl_def)
top_popc_cyl = ((total_popc_cyl_def-total_popc_cyl_def.iloc[0,:]))
#top_popc_cyl = ((total_popc_cyl_def-total_popc_cyl_def.iloc[0,:])/total_popc_cyl_def.iloc[0,:])*100
total_popc_junc_def = pd.DataFrame(((total_popc.iloc[:, 8:24].sum(axis=1))))
#total_popc_junc_def = pd.DataFrame((10534-(total_popc.iloc[:, 8:24].sum(axis=1))))
#top_popc_junc = ((total_popc_junc_def-total_popc_junc_def.iloc[0,:]))
top_popc_junc = ((total_popc_junc_def-total_popc_junc_def.iloc[0,:]))
#top_popc_junc = ((total_popc_junc_def-total_popc_junc_def.iloc[0,:])/total_popc_junc_def.iloc[0,:])*100
total_popc_flat_def = pd.DataFrame(((total_popc.iloc[:, 24:].sum(axis=1))))
#top_popc_flat = ((total_popc_flat_def-total_popc_flat_def.iloc[0,:])
top_popc_flat = ((total_popc_flat_def-total_popc_flat_def.iloc[0,:]))
#top_popc_flat = ((total_popc_flat_def-total_popc_flat_def.iloc[0,:])/total_popc_flat_def.iloc[0,:])*100



######################with respect to first frame----not working-----
#total_popc_cyl_def = pd.DataFrame(total_popc.iloc[:, 0:8].sum(axis=1))
#print ('checking new cylinder regions')
#print (total_popc_cyl_def)
#top_popc_cyl = ((total_popc_cyl_def-total_popc_cyl_def.iloc[0,:])/total_popc_cyl_def.iloc[0,:])*100
#total_popc_junc_def = pd.DataFrame(total_popc.iloc[:, 8:24].sum(axis=1))
#top_popc_junc = ((total_popc_junc_def-total_popc_junc_def.iloc[0,:])/total_popc_junc_def.iloc[0,:])*100
#total_popc_flat_def = pd.DataFrame(total_popc.iloc[:, 24:].sum(axis=1))
#top_popc_flat = ((total_popc_flat_def-total_popc_flat_def.iloc[0,:])/total_popc_flat_def.iloc[0,:])*100
############original####
#top_popc_junc = (((total_popc.iloc[:, 8:24]-total_popc.iloc[0,:])/total_popc.iloc[0,:]).sum(axis=1))
#top_popc_flat = (((total_popc.iloc[:, 24:]-total_popc.iloc[0,:])/total_popc.iloc[0,:]).sum(axis=1))
np.savetxt('top_popc_count.txt',(10297-total_popc.sum(axis=1)))
np.savetxt('top_popc_cyl.txt',top_popc_cyl)
np.savetxt('top_popc_junc.txt',top_popc_junc)
np.savetxt('top_popc_flat.txt',top_popc_flat)
np.savetxt('top_popc_cyl_def.txt',total_popc_cyl_def)
np.savetxt('top_popc_junc_def.txt',total_popc_junc_def)
np.savetxt('top_popc_flat_def.txt',total_popc_flat_def)
print (PC_fract)

TS_fract = np.zeros((n_frames, n_bins))
test_bin = np.zeros((n_frames, n_bins))
for i in range(digitized_coords.min(), digitized_coords.max() + 1):
    digitized_mask = (digitized_coords == i)
    TS_fract[:, i - 1] = digitized_mask[:, TEST].sum(axis=1) / digitized_mask.sum(axis=1)
    test_bin[:, i-1]=(digitized_mask[:, TEST].sum(axis=1))
np.savetxt('test_bin.txt', test_bin)
total_test =pd.DataFrame(test_bin/2496)
print (total_test.sum(axis=1))
row=np.arange(len(total_test))
rownum=np.squeeze(row).shape
np.savetxt('top_test_count.txt',(2496-total_test.sum(axis=1)))
#top_test_cyl = (((total_test.iloc[:, 0:8]-total_test.iloc[0,:])/total_test.iloc[0,:]).sum(axis=1))
#top_test_junc =(((total_test.iloc[:, 8:24]-total_test.iloc[0,:])/total_test.iloc[0,:]).sum(axis=1))
#top_test_flat = (((total_test.iloc[:, 24:]-total_test.iloc[0,:])/total_test.iloc[0,:]).sum(axis=1))

total_test_cyl_def = pd.DataFrame(((total_test.iloc[:, 0:8].sum(axis=1))))
print ('checking new cylinder regions-test')
print (total_test_cyl_def)
top_test_cyl = ((total_test_cyl_def-total_test_cyl_def.iloc[0,:]))
#top_test_cyl = ((total_test_cyl_def-total_test_cyl_def.iloc[0,:])/total_test_cyl_def.iloc[0,:])*100
total_test_junc_def = pd.DataFrame(((total_test.iloc[:, 8:24].sum(axis=1))))
top_test_junc = ((total_test_junc_def-total_test_junc_def.iloc[0,:]))
#top_test_junc = ((total_test_junc_def-total_test_junc_def.iloc[0,:])/total_test_junc_def.iloc[0,:])*100
total_test_flat_def = pd.DataFrame(((total_test.iloc[:, 24:].sum(axis=1))))
top_test_flat = ((total_test_flat_def-total_test_flat_def.iloc[0,:]))
#top_test_flat = ((total_test_flat_def-total_test_flat_def.iloc[0,:])/total_test_flat_def.iloc[0,:])*100
np.savetxt('time.txt',row*0.0002)
np.savetxt('top_test_cyl.txt',top_test_cyl)
np.savetxt('top_test_junc.txt',top_test_junc)
np.savetxt('top_test_flat.txt',top_test_flat)
np.savetxt('top_test_cyl_def.txt',total_test_cyl_def)
np.savetxt('top_test_junc_def.txt',total_test_junc_def)
np.savetxt('top_test_flat_def.txt',total_test_flat_def)
print (TS_fract)


np.savetxt('top_PC-frac.txt', PC_fract)
np.savetxt('top_TS-frac.txt', TS_fract)


minor_component_fraction = PC_fract+TS_fract

print (minor_component_fraction)

dt = 200 #####change dt values as your wish time interval
print (minor_component_fraction)

#first = np.array(minor_component_fraction[0,:])
#print (first)
#x=bin_rho
#y=bin_z


if dt > 0:
    newdims = [*PC_fract.shape]
    newdims[0] = int(np.floor(float(PC_fract.shape[0]) / dt))
    reduced_array = np.zeros(newdims)
    for i in range(0, newdims[0]):
        reduced_array[i, :] = PC_fract[i * dt : (i + 1) * dt, :].mean(axis=0)
    PC_fraction = reduced_array
print (PC_fraction)
np.savetxt('PC_fraction_top.txt', PC_fraction)
np.savetxt('PC_fraction_mean_top.txt', PC_fraction.mean(axis=0))
print (PC_fraction.mean(axis=0))


if dt > 0:
    newdims = [*TS_fract.shape]
    newdims[0] = int(np.floor(float(TS_fract.shape[0]) / dt))
    reduced_array = np.zeros(newdims)
    for i in range(0, newdims[0]):
        reduced_array[i, :] = TS_fract[i * dt : (i + 1) * dt, :].mean(axis=0)
    TS_fraction = reduced_array
print (TS_fraction)
np.savetxt('TS_fraction_top.txt', TS_fraction)
np.savetxt('TS_fraction_mean_top.txt', TS_fraction.mean(axis=0))
print (TS_fraction.mean(axis=0))


labels=[]
labels=[n*0.16 for n in range (1,26)]
label2 = list(map("{:.1f}".format, labels))
print (labels)


#num_ticks = 25
## the index of the position of yticks
#yticks = np.linspace(0, 26, num_ticks, dtype=np.int)
## the content of labels of these yticks
#ytickls = [depth_list[idx] for idx in yticks]

df1 = pd.DataFrame(PC_fraction)
df1_trans = df1.transpose()
ax = sns.heatmap(df1_trans, xticklabels=label2, yticklabels=format_bins, center=0.8, cmap='seismic')
ax.invert_yaxis()
plt.axhline(y=8, ls='--', c='black')
plt.axhline(y=24, ls='--', c='black')
#plt.xticks(ax.get_xticks(), ax.get_xticks() * 2)
ax.xlabel="20 ns bin"
plt.ylabel="Transformed mito axis (nm)"
plt.xticks(rotation=30)
plt.title("PT-PC-fraction_top")
plt.show()

df2 = pd.DataFrame(TS_fraction)
df2_trans = df2.transpose()
ax = sns.heatmap(df2_trans, xticklabels=label2, yticklabels=format_bins, center=0.2, cmap='seismic')
ax.invert_yaxis()
plt.axhline(y=8, ls='--', c='black')
plt.axhline(y=24, ls='--', c='black')
plt.xlabel="20 ns bin"
plt.ylabel="Transformed mito axis (nm)"
plt.xticks(rotation=30)
plt.title("PT-TS-fraction_top")
plt.show()

####plotting 4 figs together 
''' fig, axn = plt.subplots(2, 2, sharex=True, sharey=True)
cbar_ax = fig.add_axes([.91, .3, .03, .4])

for i, ax in enumerate(axn.flat):
    sns.heatmap(df1, ax=ax,
                cbar=i == 0,
                vmin=0, vmax=1,
                cbar_ax=None if i else cbar_ax)   '''

