# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:53:44 2015

@author: jaguirre
Solve a recurring problem I have of finding the days on which classes fall.
""" 

from dateutil import rrule
from dateutil.rrule import *
from datetime import datetime, timedelta

def slashdate(date):
    return str(date.month)+'/'+str(date.day)+'/'+str(date.year)[-2:]
    
def longdate(date):
    ctime = date.ctime()
    return ctime[0:10]+ctime[-5:]
    
def courseDates(startdate, enddate, excludedates, byweekday):
    
    dates = rrule(DAILY, dtstart=startdate, until=enddate, byweekday=byweekday)
    gooddates = []
    
    for date in dates:
        gooddate = True
        for exclude in excludedates:
            if date == exclude:
                gooddate = False
        if gooddate:
            gooddates.append(date)

    for i,dt in enumerate(gooddates):
        prtstr = 'Class '+str(i+1)+' ('+slashdate(dt)+')'

        print(prtstr)
    
    print('Number of class meetings', len(gooddates))