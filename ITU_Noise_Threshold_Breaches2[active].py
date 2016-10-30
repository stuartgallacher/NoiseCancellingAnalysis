from matplotlib import pyplot as plt
from pylab import plot, show, savefig, xlim, figure, hold, ylim, legend, boxplot, setp, axes 
import numpy as np
import scipy.stats as sp
import time

'''a further supplementary piece of code that borrows from the main file. It will iterate through the directory of the
full set of ten day data and from that we will scrape the absolute number of spikes per minute of the day in each group 
so we determine whether or not we are suppressing the number of breaches per second'''
control_breaches = []; nchoff_breaches=[]; nchon_breaches=[]
control_breaches_std =[]; nchoff_breaches_std=[]; nchon_breaches_std = []

o = open('noise_data_10_days.csv', 'r')
p = o.readlines()


def NoiseThresholdBreachesPerMinute(p):
	sli_control = {}; sli_nchoff = {}; sli_nchon ={}
	sli_control_full={}

	def threshold_filter(t):
		counter = 0
		for i in t: 
			for i in i:
				i = float(i)
				if i >55:
					counter += 1
		return counter		#number of seconds in the 10 second grouping that breeches the threshold	

	for i in p[1:]:
		n = i.split('\n')
		for i in n:
			t = i.split('\t')
			if len(t) > 1:
				dt = time.strptime(t[30], ' %d-%m-%Y %H:%M:%S')
				con_li = [t[0:9]]
				nchoff_li = [t[10:19]]
				nchon_li = [t[20:29]]
				
				filtered_t_con = threshold_filter(con_li)
				filtered_t_nchoff = threshold_filter(nchoff_li)
				filtered_t_nchon = threshold_filter(nchon_li)

				sli_control[dt.tm_hour, dt.tm_min, dt.tm_sec] = filtered_t_con
				sli_nchoff[dt.tm_hour, dt.tm_min, dt.tm_sec] = filtered_t_nchoff
				sli_nchon[dt.tm_hour, dt.tm_min, dt.tm_sec] = filtered_t_nchon
	

	return sli_control, sli_nchoff, sli_nchon

def MinuteMaker(sli, x):
	li =[]; 
	for i, j in (sli.items()):		
		if i[0] == x:
			li.append(j)
			std = sp.sem(li)
	return sum(li)/60, 	std
		

#	fig = plt.figure()
#	ax = axes()			
#	plt.plot_date(sli_control.keys(), sli_control.(), fmt='b-', tz=None, xdate =True, ydate=False)
#	#plt.plot_date(sli_nchoff.values(), sli_nchoff.keys(), fmt='g-', tz=None, xdate=True, ydate=False)
#	#plt.plot_date(sli_nchon.values(), sli_nchon.keys(), fmt='r-', tz=None, xdate=True, ydate= False)
#
#	plt.show()



		#0-9 control
		#10-19 off
		#20-29 on
		#30 datetime


a,b,c = NoiseThresholdBreachesPerMinute(p)
for i in range(0, 24):
	d, std = MinuteMaker(a, i)
	control_breaches.append(d)
	control_breaches_std.append(std)

for i in range(0,24):
	d, std = MinuteMaker(b, i)
	nchoff_breaches.append(d)
	nchoff_breaches_std.append(std)

for i in range(0, 24):
	d, std = MinuteMaker(c, i)
	nchon_breaches.append(d)
	nchon_breaches_std.append(std)


N = 24
ind = np.arange(N)
width = 0.35


fig, ax = plt.subplots()
bars1 = ax.bar(ind, control_breaches, width, color='#ffffff', yerr=control_breaches_std, label = 'control')
bars2 = ax.bar(ind+width, nchoff_breaches, width, color='#999999', yerr=nchoff_breaches_std, label = 'NCH off')
bars3 = ax.bar(ind+(width*2), nchon_breaches, width, color='#000000', yerr=nchon_breaches_std, label = 'NCH on')
ax.set_ylabel('mean number of breaches per minute per hour (>55 Db)')
ax.set_xticks(ind+(width*2))
ax.set_xticklabels(('00:00','','','','','','06:00','','','','','','12:00','','','','','','18:00','','','','',''))
ax.legend(loc='upper right')

plt.savefig('pub_threshold_breaches', format='png', dpi=1200)


o.close()
