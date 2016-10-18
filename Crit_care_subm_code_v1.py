from matplotlib import pyplot as plt
from pylab import plot, show, savefig, xlim, figure, hold, ylim, legend, boxplot, setp, axes 
import numpy as np
import scipy.stats as sp
from datetime import datetime
import time
import itertools
import operator
from scipy.interpolate import spline



#-----------------------------------#
#Author Dr Stuart Gallacher GMC 7421022
#Plymouth Hospitals NHS Trust
#Email stuartgallacher@nhs.net


#Copyright 2015 Stuart Gallacher 
#Distributed under the  terms of the GNU general public licence v3
#-----------------------------------#


'''a general purpose script for handling the creation of graphs, data wrangling and statistical analysis.  PlottingClass outputs:
histogram, various forms of timeslicing, average per second noise exposure for each hour of the day. StatsClass outputs: results of
ANOVA and t-testing.'''

o = open('noise_averages_file.csv', 'r') #Input for PlottingClass and StatsClass
p = o.readlines()
	
l = open('noise_data_10_days.csv', 'r') #Input for BreachesClass
m = l.readlines()

class PlottingClass:
	'''A class that takes a readlines file and creates hourly box plots or histograms.'''

	def __init__(self, p):
		self.p= p

	def PlotHistogram(self,p):
		'''creates an overlay histogram of the groups noise output.'''	
		li_control = []; li_nchoff = []; li_nchon = [] #0, 3, 6
		for i in p[2:]:
				i = (i.split('\t'))
				li_control.append(float(i[0])) 
				li_nchoff.append(float(i[3]))
				li_nchon.append(float(i[6]))

		ax = axes()		
		bins = np.linspace(0, 100, 75)

		plt.plot((35,35),(0, 25000), 'r-', linewidth = 2.0) 
		plt.hist(li_control, bins, alpha=0.3, label='control')
		plt.hist(li_nchoff, bins, alpha=0.3, label='NCH off')
		plt.hist(li_nchon, bins, alpha=0.3, label='NCH on')
		ax.set_xlabel('Sound Intensity (Db)')
		ax.set_ylabel('Frequency')
		plt.legend(loc='upper right')
		plt.show()


	def TimeDataWrangle(self, p):

		'''a function that takes the data and cuts it into the different time groups we need (i.e hours and minutes).
		Does not output to file.'''

		mli_control = {} #minute level dictionary of noise 
		mli_nchon = {} 
		hour_control = {} #hour level dictionary
		hour_nchon = {}

		for i in p[2:]:
			i = i.split('\t')
			t = time.strptime(i[1], ' %d-%m-%Y %H:%M:%S')
			try:
				mli_control[t.tm_hour, t.tm_min] += float(i[0])
				mli_nchon[t.tm_hour, t.tm_min] += float(i[6])
			except KeyError:
				mli_control[t.tm_hour, t.tm_min] = float(i[0])
				mli_nchon[t.tm_hour, t.tm_min] = float(i[6])

		for i,j in mli_control.items():
			try:
				hour_control[i[0]] += j
			except KeyError:
				hour_control[i[0]] = j

		for i, j in mli_nchon.items(): 
			try:
				hour_nchon[i[0]] += j
			except KeyError:
				hour_nchon[i[0]] = j

		return mli_control, mli_nchon, hour_control, hour_nchon

	def PlotPrepFunct(self, tsd, tsd2): #time specific dict
		'''takes an minute level dictionary and returns dictionary of the noise level in each hour.'''	
		li = []; li2 =[]
		group_li = []; group_li2 = []

		for i,j in sorted(tsd.items()):
			mindata_tuple =  (i[0], j)
			li.append(mindata_tuple)
			
		for key, group in itertools.groupby(li, operator.itemgetter(0)):
				 group_li.append(np.array(list(x[1] for x in group)))
		
		for i,j in sorted(tsd2.items()):
			mindata_tuple =  (i[0], j)
			li2.append(mindata_tuple)
			
		for key, group in itertools.groupby(li2, operator.itemgetter(0)):
				 group_li2.append(np.array(list(x[1] for x in group)))		 	

		return group_li, group_li2

		
	def PlotLineGraph(self, a, b):

		y1=[]; y2 =[]
		x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
		for i in a:
			n = i/60 
			y1.append(n)

		for i in b:
			n = i/60
			y2.append(n)	

		yerr1pos=[];yerr1neg=[]; yerr2pos = [];yerr2neg=[]

		def mean_confidence_interval(data, confidence=0.95):
			a = 1.0*np.array(data)
			n = len(a)
			m,se = np.mean(a), sp.sem(a)
			h = se*sp.t._ppf((1+confidence)/2, n-1)
			return m+h, m-h
		

		my1 = np.array([np.mean(i) for i in y1])
		my2 = np.array([np.mean(i) for i in y2])
		x_sm = np.array(x)

		x_smooth = np.linspace(x_sm.min(),x_sm.max(),200)
		

		for i in y1:
			q,r= mean_confidence_interval(i)
			yerr1pos.append(q)
			yerr1neg.append(r)

		for i in y2:
			q,r = mean_confidence_interval(i)
			yerr2pos.append(q)	
			yerr2neg.append(r)

		yerr1pos = np.array(yerr1pos)
		yerr1neg = np.array(yerr1neg)

		y1_smooth = spline(x, my1, x_smooth)
		y2_smooth = spline(x, my2, x_smooth)
		yerr1pos_smooth = spline(x, yerr1pos, x_smooth)
		yerr1neg_smooth = spline(x, yerr1neg, x_smooth)
		yerr2pos_smooth = spline(x, yerr2pos, x_smooth)
		yerr2neg_smooth = spline(x, yerr2neg, x_smooth)

		fig = plt.figure()
		ax = axes()

		plt.plot(x_smooth, y1_smooth, color='#808080',linewidth=1)
		plt.plot(x_smooth, yerr1pos_smooth, color='#808080', linestyle='--')
		plt.plot(x_smooth, yerr1neg_smooth, color='#808080', linestyle='--')
		plt.plot(x_smooth, y2_smooth, 'b-', linewidth = 1)
		plt.plot(x_smooth, yerr2pos_smooth, color='#808080', linestyle='--')
		plt.plot(x_smooth, yerr2neg_smooth, color='#808080', linestyle='--')
		ax.set_xticklabels(['00:00','05:00','10:00','15:00','20:00'])
		ax.set_ylabel('Average Noise Exposure Per Second (Db)')
		ax.set_xlim([0,23])	

		plt.show()	



class StatsClass:
	'a class that build on the work above, taking the prepared data and running the various required statistical tests'

	def __init__(self, s):
		self.s = s


	def OneWayAnova(self, p):	
		'a one way anova comparing if there is a significant difference between the means of the three groups. Can use the groups produced to t-test'
		c =[]; nchoff = []; nchon = []
		for i in p[2:]:
			i = i.split('\t')
			c.append(i[0])
			nchoff.append(i[3])
			nchon.append(i[6])
		npc = np.array(c)
		npnchoff = np.array(nchoff)
		npnchon = np.array(nchon)
		anova = sp.f_oneway(c, nchon, nchoff)
		print anova #(statistic=113103.90190322432, pvalue=0.0)

	def TwoWayPairedTTesting(self, p):
		'running the various t-tests'
		c =[]; nchoff = []; nchon = []
		for i in p[2:]:
			i = i.split('\t')
			c.append(float(i[0]))
			nchoff.append(float(i[3]))
			nchon.append(float(i[6]))
		#npc = np.array(c)
		#npnchoff = np.array(nchoff)
		#npnchon = np.array(nchon)
		#con_vs_nchoff = sp.ttest_rel(c, nchoff) #T-statistic=413.68750052572483, pvalue=0.0 (<0.0001)
		#con_vs_nchon = sp.ttest_rel(c, nchon) #T-statistic=1004.8657304590413, pvalue=0.0) (<0.0001)
		#nchoff_vs_nchon = sp.ttest_rel(nchoff, nchon) #T-statistic=583.3248485420803, pvalue=0.0 (<0.0001)
		#print nchoff_vs_nchon
		print np.mean(c), np.std(c) # 51.3
		print np.mean(nchoff), np.std(nchon) #
		print np.mean(nchon), np.std(nchon) #


class BreachesClass:
	'A class to contain the breaches graph preparation'
	control_breaches = []; nchoff_breaches=[]; nchon_breaches=[]
	control_breaches_std =[]; nchoff_breaches_std=[]; nchon_breaches_std = []


	def  __init__(self, m):
		self.m = m

	def noise_threshold_breaches_per_minute(p):
		sli_control = {}; sli_nchoff = {}; sli_nchon ={}
		sli_control_full={}

		def threshold_filter(t):
			counter = 0
			for i in t: 
				for i in i:
					i = float(i)
					if i >55:
						counter += 1
			return counter		

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
			


	a,b,c = noise_threshold_breaches_per_minute(m)
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
	bars1 = ax.bar(ind, control_breaches, width, color='#ffffff', yerr=control_breaches_std)
	bars2 = ax.bar(ind+width, nchoff_breaches, width, color='#999999', yerr=nchoff_breaches_std)
	bars3 = ax.bar(ind+(width*2), nchon_breaches, width, color='#000000', yerr=nchon_breaches_std)
	ax.set_ylabel('mean number of breaches per minute per hour (>55 Db)')
	ax.set_xticks(ind+(width*2))
	ax.set_xticklabels(('00:00','','','','','','06:00','','','','','','12:00','','','','','','18:00','','','','',''))

	plt.show()	



###Plotting Class Initators
#c = PlottingClass(p) #- to start the class
#mli_control, mli_nchon, hour_control, hour_nchon = c.TimeDataWrangle(p) #to start this aux function
#a,b = c.PlotPrepFunct(mli_control, mli_nchon) #- to prep data for boxplots
#c.PlotLineGraph(a,b) - to initiate line graph
#c.PlotHistogram(p)  - to initiate histogram

### Stats Class Initiators
#s = StatsClass(p)
#so = s.OneWayAnova(p)
#s.TwoWayPairedTTesting(p)


###Breaches Class Initiators
b = BreachesClass(m) #to start the class

o.close()
l.close()

