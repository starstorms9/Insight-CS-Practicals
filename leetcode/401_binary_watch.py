class Solution:
    def readBinaryWatch(self, num: int):
        times = []
        digits = 10
        
        binaries = list(range(digits))
        for i in range(digits) :
            binaries[i] = 2 ** (i)
        binaries = list(reversed(binaries))
        
        for i in range(2 ** digits) :
            sbin, count = self.binarize(i, binaries, digits)
            if count == num :         
                time, valid = self.getString(sbin)
                if valid :
                    times.append(time)
        return times

    def getString(self, binstring) :
        hours = binstring[:4]
        mins = binstring[4:]
        
        hourNum = self.unBinarize(hours)
        minNum = self.unBinarize(mins)
        
        if hourNum >= 13 or minNum >= 60 :
            return '', False
        
        timeStr = "{}:{:02d}".format(hourNum, minNum)
        return timeStr, True
    
    def unBinarize(self, bstring) :
        number = 0
        for i in range(len(bstring)) :
            if bstring[i-1] == '1' :
                number += 2 ** i
        return number
    
    def binarize(self, num, binaries, digits) :
        s = []
        rem = num
        count = 0
        for i in range(digits) :
            if rem >= binaries[i] :
                s.append('1')
                rem -= binaries[i]
                count += 1
            else :
                s.append('0') 
        return ''.join(s), count
        
#%%
sol = Solution()
sol.readBinaryWatch(2)

#%%
import itertools

LEDS_TO_HOURS_AND_MINUTES = [
    (8, 0),
    (4, 0),
    (2, 0),
    (1, 0),
    (0, 32),
    (0, 16),
    (0, 8),
    (0, 4),
    (0, 2),
    (0, 1),
]


def leds_to_tuple(leds_on):
    """ given an iterable of LED indices """
    hours, minutes = 0, 0
    for led_index in leds_on:
        led_hours, led_minutes = LEDS_TO_HOURS_AND_MINUTES[led_index]
        hours += led_hours
        minutes += led_minutes
    return hours, minutes


def is_valid_time(hours_and_minutes):
    """ Return whether the hours and minutes provided represent a legit time on a 12-hour watch
    note: acceptable times are between 0:00 and 11:59
    """
    hours, minutes = hours_and_minutes
    return 0 <= hours <= 11 and 0 <= minutes < 60


def format_time(hours_and_minutes):
    hours, minutes = hours_and_minutes
    return f"{hours}:{minutes:02}"

def readBinaryWatch(num):
    led_indices = range(len(LEDS_TO_HOURS_AND_MINUTES))
    for leds_on in itertools.combinations(led_indices, num):
        
        hours_and_minutes = leds_to_tuple(leds_on)
        if is_valid_time(hours_and_minutes):
            yield format_time(hours_and_minutes)
                
#%%
times = list(readBinaryWatch(2))
print(times)

#%%
combs = itertools.combinations(list(range(10)), 2)
combsl = list(combs)
