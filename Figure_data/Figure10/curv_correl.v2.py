import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import stats
from scipy.stats import pearsonr
plt.rcParams.update({'font.size':18})
plt.rcParams.update({'figure.autolayout': True})

plt.ion()
#POPC/POPE - POPE
## Order Outer-Junc, outer-cyl, outer-flat, inner-junc, inner-cyl, inner-flat
mean10 = np.array([-3.16, 4.22, -1.48, 0.68, -6.24, 0.94])*10**-2
mean15 = np.array([-4.20, 3.04, -2.51, 1.71, -4.10, 1.48])*10**-2
mean5 = np.array([-1.00, 7.43, -0.95, -1.50, -10.73, 0.43])*10**-2

#mean = np.array([4.26, -3.4, -6.5,0.1, -1.3, 0.3])*10**-2
mean10_mean = np.mean(mean10)
mean5_mean = np.mean(mean5)
mean15_mean = np.mean(mean15)

mean10_sd = np.std(mean10)
mean5_sd = np.std(mean5)
mean15_sd = np.std(mean15)


gauss10 = np.array([-6.52, -0.12, 0, -4.90, 0.27, 0.25])*10**-3
gauss15 = np.array([-4.5, 0, 0.1, -3.6, 0.2, 0.2])*10**-3
gauss5 = np.array([-10.4, 0.2, 0, -9.7, -0.8, -0.9])*10**-3
#gauss = np.array([-.1,-6.3,0.3,-5.4,0.1,0.2])*10**-3
gauss10_mean = np.mean(gauss10)
gauss5_mean = np.mean(gauss5)
gauss15_mean = np.mean(gauss15)

gauss10_sd = np.std(gauss10)
gauss5_sd = np.std(gauss5)
gauss15_sd = np.std(gauss15)


enr = np.zeros((13,6))
enr[0,:]=[2.7, -2.1, -0.9, -1.8, 2.5, 0.2] # POPE
enr[1,:]= [2.9, -0.9, -1.2, -0.8, 2.5, -0.8] #  DOPE
enr[2,:]= [3, -3.4, 0.1, -1.7, 3.3, 0.2] # CDL
enr[3,:] = [1.5, -1.8, -0.1, -1.5, 1.1, 0.5]  #POPE with CDL-2
enr[4,:] = [2.7, -3, -0.2, -2.4, 3, 0.6] #CDL-2 with POPE
enr[5,:] = [2, -1.1, -0.7, -1.4, 0.7, 0.8] # DOPE with CDL-2
enr[6,:] = [2.1, -0.9, -0.8, -0.1, 2.5, -0.8] # CDL-2 with DOPE
enr[7,:] = [1.9, -1.9, -0.1, -0.7, 1.9, -0.2] # POPE with CDL-1
enr[8,:] = [4.3, -3.5, -0.7, -0.7, 3.1, -0.7] # CDL-1 with POPE
enr[9,:] =  [2.1, -1.3, -1.6, -0.5, 1.1, -0.2] # POPE 15 nm system
enr[10,:] = [3.1, -2.5, -1.3, -1.1, 2.1, -0.4] # CDL-2 15 nm system
enr[11,:] = [1.8, -1.6, -0.2, 0.2, 1.2, -0.2]  # POPE 5 nm system
enr[12,:] = [3.2, -4.7, 0.1, -1.1, 4.5, -0.3] # CDL-2 5 nm system
#enr[0,:] =[-4.3,-2.7,3.8,-2.1,1.6,-1.6]  # POPE
#enr[1,:] = [-3.4, 4.6, -1.2, 4.1, -1.9, -2.2]  # DOPE
#enr[2,:] = [-5.2, 5.7, -0.4, 2.9, -2.5, -0.4]  # CDL
#enr[3,:] = [-2.6, 2.4, 0.2, 1.6, -3.7, 2.1]  #POPE with CDL-2
#enr[4,:] = [-5.0, 4.0, 1.0, 2.7, -4.9, 2.2]  # CDL-2 with property
#enr[5,:] = [-3.3, 2.4, 0.9, 3.8, -5.5, 1.7] # DOPE with CDL-2
#enr[6,:] = [-5.0, 3.3, 1.7, 4.1, -5.5, 1.4]  # CDL-2 with DOPE
#enr[7,:] = [-1.1, 1.2, -0.1, 1.9, -1.6, -0.3] # POPE with CDL-1
#enr[8,:] = [-4.9, 5.4, -0.5, 5.7, -5.8, 0.1]  # CDL-1 with POPE


ncond = np.shape(enr)[0]
pv_m = np.zeros(ncond)
pv_g = np.zeros(ncond)
corr_m = np.zeros(ncond)
corr_g = np.zeros(ncond)


for i in range(0,len(enr)):
    ind = np.argsort(enr[i,:])
    enr_s = enr[i,ind]
    mean10_s = (mean10[ind] - mean10_mean)/mean10_sd   ### Standarized inputs
    gauss10_s = (gauss10[ind] - gauss10_mean)/gauss10_sd
    mean5_s = (mean5[ind] - mean5_mean)/mean5_sd   ### Standarized inputs
    gauss5_s = (gauss5[ind] - gauss5_mean)/gauss5_sd
    mean15_s = (mean15[ind] - mean15_mean)/mean15_sd   ### Standarized inputs
    gauss15_s = (gauss15[ind] - gauss15_mean)/gauss15_sd

    #curv = [mean_s,gauss_s]


    # plt.figure()
    # popt_mean, pcov_mean = sc.optimize.curve_fit(funca, mean_s, enr_s)
    # perr_mean = np.sqrt(np.diag(pcov_mean))
    # residuals_mean = enr_s- funca(mean_s, *popt_mean)
    # ss_res_mean = np.sum(residuals_mean**2)
    # ss_tot = np.sum((enr_s-np.mean(enr_s))**2)
    # r_squared_mean[i] = 1 - (ss_res_mean / ss_tot)
    #
    # popt_gau, pcov_gau = sc.optimize.curve_fit(funca, gauss_s, enr_s)
    # perr_gau = np.sqrt(np.diag(pcov_gau))
    # residuals_gau = enr_s- funca(gauss_s, *popt_gau)
    # ss_res_gau = np.sum(residuals_gau**2)
    # r_squared_gau[i] = 1 - (ss_res_gau / ss_tot)
    #
    # popt_both, pcov_both = sc.optimize.curve_fit(funcb, curv,enr_s)
    # perr_both = np.sqrt(np.diag(pcov_both))
    # residuals_both = enr_s- funcb(curv, *popt_both)
    # ss_res_both = np.sum(residuals_both**2)
    # r_squared_both[i] = 1 - (ss_res_both / ss_tot)
    #
    #
    #
    # ax = plt.axes(projection='3d')
    # ax.scatter(mean_s,gauss_s,enr_s)
    # ax.plot(mean_s,gauss_s,funcb(curv,*popt_both))


    # fit[i,:] = popt_both
    # fit_err[i,:] = perr_both
    if i < 9:
        corr_m[i], pv_m[i] = pearsonr(enr_s, mean10_s)
        corr_g[i], pv_g[i]= pearsonr(enr_s, gauss10_s)
        print(i, '10 nm')
    elif i <11:
        corr_m[i], pv_m[i] = pearsonr(enr_s, mean15_s)
        corr_g[i], pv_g[i]= pearsonr(enr_s, gauss15_s)
        print(i, '15 nm')
    else:
        corr_m[i], pv_m[i] = pearsonr(enr_s, mean5_s)
        corr_g[i], pv_g[i]= pearsonr(enr_s, gauss5_s)
        print(i, '5 nm')

fig,ax = plt.subplots(figsize=(10,8))

plt.plot(corr_m, 'o', label='$r_P^{mean}$', markersize=8)
plt.plot(corr_g, 'o', color='r', label='$r_P^{Gauss}$', markersize=8)
ax.set_xticks(list(range(0,ncond)))
ax.set_xticklabels(['POPE(POPC)','DOPE(POPC)', 'CDL$^{-2}$(POPC)', 'POPE(POPC/CDL$^{-2}$)', 'CDL$^{-2}$(POPC/POPE)',\
'DOPE(POPC/CDL$^{-2}$)', 'CDL$^{-2}$(POPC/DOPE)', 'POPE(POPC/CDL$^{-1}$)', 'CDL$^{-1}$(POPC/POPE)', \
'POPE(POPC/CDL$^{-2}$, $r_{cyl}$=15 nm)', 'CDL$^{-2}$(POPC/POPE, $r_{cyl}$=15 nm)', 'POPE(POPC/CDL$^{-2}$, $r_{cyl}$=5 nm)', 'CDL$^{-2}$(POPC/POPE, $r_{cyl}$=5 nm)'], )
plt.xticks(rotation=90)
plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
plt.legend(loc=1)
plt.ylabel('$r_P$')
ax.set_yticks([0, -.2, -.4, -.6, -.8, -1.0])
ax.tick_params(labelsize=12)

plt.savefig('curv_correl.v2.tiff')
