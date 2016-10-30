from matplotlib import pyplot as plt
from pylab import plot, show, savefig, xlim, figure, hold, ylim, legend, boxplot, setp, axes
import matplotlib.mlab as mlab
import numpy as np
import scipy.stats as sp


o = open('control_vs_head.csv', 'r')
p = o.readlines()

def ControlVsHeadHistogram(p):

	li_control= []; li_head = []	
	for i in p[2:]:
		i = i.split('\t')
		li_control.append(float(i[0]))
		li_head.append(float(i[1]))
		#control_array = np.array(li_control)
		#head_array = np.array(li_head)


	#num_bins = 50
	#mu, sigma = sp.norm.fit(li_control)
	#mu2, sigma2 = sp.norm.fit(li_head)
	#n, bins, patches = plt.hist(li_control, num_bins, normed=1, alpha = 0.3, label='control')
	#y = mlab.normpdf(bins, mu, sigma)
	#plt.plot(bins, y, 'b-', linewidth= 1)
	#n2,bins2, patches2= plt.hist(li_head, num_bins, alpha=0.3, normed=1, label= 'head')
	#z = mlab.normpdf(bins2, mu2, sigma2)
	#plt.plot(bins2, z, 'g-', linewidth=1)
	#plt.xlabel('Noise Intensity (Db)')
	#plt.ylabel('Frequency')
	#plt.legend(loc='upper right')

	#plt.show()

	#signf = sp.ttest_rel(li_control, li_head)
	#print signf
	### output T-stat 230.499, p-value <0.0001 - extremely statsistically significant!!
	print np.std(li_control)
	print np.std(li_head)

#mean control 55.4

o.close()
c = ControlVsHeadHistogram(p)