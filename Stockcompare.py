import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Import Data
temp1 = np.loadtxt("barrick.txt",dtype='str',skiprows=1,unpack=True) #Barrick Gold data
temp2 = np.loadtxt("gold.txt",dtype="str",skiprows=1,unpack=True)
dateraw = np.loadtxt("date.txt",dtype="str",skiprows=1,unpack=True)

datef=dateraw.flatten()
date=datef.tolist()

barrickraw=temp1[3]
goldraw=temp2[1]

barrick=barrickraw.astype(np.float)
gold=goldraw.astype(np.float)
gold=gold[0:len(gold)-32]

x=np.arange(0.0,len(gold),1.0)
dt=1.0

#raw data
plt.figure(1, figsize=(13.66,7.68), dpi=80)
plt.plot(x,gold,label='Gold Price')
plt.ylabel("Price [USD]")
plt.xticks(np.linspace(0,len(date),18), date, rotation=45)
plt.show()

plt.figure(2, figsize=(13.66,7.68), dpi=80)
plt.plot(x,barrick,label='Barrick Gold')
plt.ylabel("Price [USD]")
plt.xticks(np.linspace(0,len(date),18), date, rotation=45)


############TREND COMPARISON###########

#stock price wrt max price
barrickrel=barrick/max(barrick)
goldrel=gold/max(gold)

#General linear function
def Fit(x,*p):
    return p[0]*x+p[1]

guess=(1.0,1.0)

popt1, pcov1 = curve_fit(Fit, x, barrickrel, p0=guess, sigma=0.1)
barricktrend=Fit(x,*popt1) 

popt2, pcov2 = curve_fit(Fit, x, goldrel, p0=guess, sigma=0.1)
goldtrend=Fit(x,*popt2) 

plt.figure(3, figsize=(13.66,7.68), dpi=80)
plt.plot(x,barrickrel,label='Barrick Gold')
plt.plot(x,goldrel,label='Gold Price')
plt.plot(x,goldtrend,'r--' ,linewidth=2, label='Trend')
plt.plot(x,barricktrend,'r--',linewidth=2)
plt.ylabel("Relative Price (P/Pmax) [USD]")
plt.xticks(np.linspace(0,len(date),18), date, rotation=45)
plt.legend()

sloperatio=popt2[0]/popt1[0]

print "Slope Ratio (Gold/Barrick): ", sloperatio
print ''

########FILTERTING############

hann=np.zeros_like(gold)
for i in range(0,len(gold)):
    hann[i]=1.0-np.cos(2*np.pi*i/len(gold))
    
barrickfix=(barrickrel-barricktrend)*hann
goldfix=(goldrel-goldtrend)*hann

#detrend and filtered data
plt.figure(4, figsize=(13.66,7.68), dpi=80)
plt.plot(barrickfix,label='Barrick Gold')
plt.plot(goldfix,label='Gold Price')
plt.ylabel("Relative Price (P/Pmax) [USD]")
plt.xticks(np.linspace(0,len(date),18), date, rotation=45)
plt.legend()

#############CROSS-CORRELATION############

GOLD=np.fft.fft(goldfix,len(goldfix))
BAR=np.fft.fft(barrickfix,len(goldfix))

BARconj=np.conj(BAR)
GOLDconj=np.conj(GOLD)

cxy=np.fft.ifft(GOLDconj*BAR)

xcross=np.linspace(-len(cxy)/2.0,len(cxy)/2.0,len(cxy))

cxyrel=np.fft.fftshift(cxy)/max(np.real(cxy))

plt.figure(5, figsize=(13.66,7.68), dpi=80)
plt.plot(xcross,cxyrel,linewidth=2)
plt.ylabel('C$_{xy}$',size=14)
plt.xlabel("Time Difference [days]")

#finding x-value of min of cross-correlation
for i in range(0,len(cxy)):
    if cxyrel[i]>=np.max(cxyrel):
        xmax=xcross[i]
        
print 'Days shifted: ', xmax # number of days shifted

plt.plot((xmax, xmax), (-0.4, 1.0), 'k--',linewidth=3) #insert a vertical line at x s.t. f(x)
plt.show()


############SEASONAL TRENDS (FFTing)############

df=1.0/len(gold)
freq = np.fft.fftfreq(len(x),dt)
GOLDi=np.imag(GOLD)
BARi=np.imag(BAR)

#Signal to noise ratio value 
SNRgold=5.0 #extract peaks higher than 5.0
SNRbar=10.0 #extract peaks higher than 10.0

goldpeaks=[]
barpeaks=[]

for i in range(0,len(GOLDi)):
    if GOLDi[i]>=SNRgold:
        goldpeaks.append(np.abs(1.0/freq[i]))

print ''        
print 'Prominent Gold Periods: ', goldpeaks
print ''

for j in range(0,len(BARi)):
    if BARi[j]>=SNRbar:
        barpeaks.append(np.abs(1.0/freq[j]))
        
print 'Prominent Barrick Gold Periods: ', barpeaks

plt.figure(6, figsize=(13.66,7.68), dpi=80)
plt.subplot(210)
plt.plot(freq,GOLDi,label='Gold')
plt.plot((-0.05,0.05), (SNRgold, SNRgold), 'k',linewidth=2)
plt.ylabel('FFT(Gold)')
plt.xlabel('Frequency [1/days]')
plt.xlim(-0.05,0.05)
plt.legend()

plt.subplot(211)
plt.plot(freq,BARi,label='Barrick Gold')
plt.plot((-0.05,0.05), (SNRbar, SNRbar), 'k',linewidth=2)
plt.ylabel('FFT(Barrick Gold)')
plt.xlim(-0.05,0.05)
plt.legend()
plt.show()