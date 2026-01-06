#date magic
date = date.astype('datetime64[D]')
years = date.astype('datetime64[Y]') 
months= (date.astype('datetime64[M]')-date.astype('datetime64[Y]')+1).astype('int64')
weeks = (date.astype('datetime64[W]')-date.astype('datetime64[Y]')+1).astype('int64')
weekdays = (date.astype('datetime64[D]').astype('int64') -4) % 7 +1
monthdays =  date.astype('datetime64')-date.astype('datetime64[M]') + 1
yeardays = date.astype('datetime64')-date.astype('datetime64[Y]') + 1

years=np.unique(years,return_inverse=1)
months=np.unique(months,return_inverse=1)
weeks=np.unique(weeks,return_inverse=1)
weekdays=np.unique(weekdays,return_inverse=1)
monthdays=np.unique(monthdays,return_inverse=1)
yeardays=np.unique(yeardays, return_inverse=1)


