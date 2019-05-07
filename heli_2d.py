import numpy as np



class Heli2D:
    
    def __init__(self):
        self.g = 9.81
        self.cl_alpha = 5.7 # NACA0012
        self.volh = .075   # blade solidity parameter
        self.lok = 6
        self.cds = 1.5
        self.mass = 2200
        self.rho = 1.225
        self.v_tip = 200
        self.diam = 2 * 7.32
        self.iy = 10615
        self.mast = 1
        self.omega = self.v_tip / (self.diam / 2)
        self.area = np.pi / 4 * self.diam ^ 2
        self.tau = 0.1
        

hwens=25
c(i) = u(i)*sin(pitch(i)) - w(i)*cos(pitch(i))
h(i)=-z(i)
cwens(i) = .1*(hwens-h(i))
collectgrd(i) = 5 + 2*(cwens(i)-c(i)) + 0.2*corrc(i)
collect(i)=collectgrd(i)*pi/180

#LAW FOR LONGIT. CYCLIC
if t(i)<90 
    #law 1 for helic. pitch 
    uwens=50
    pitchwens(i)=-.005*(uwens-u(i))-.0005*corr(i)	#in rad
    xeind(i)=x(i)
    pitcheind(i)=pitchwens(i)
else
    #law 2 for helic.pitch 
    xxeind=xeind(900)
    pitcheeind=pitcheind(900)
    pitchwens(i)=-.001*(xxeind+2000-x(i))+.02*u(i)	#in rad
    if pitchwens(i)<pitcheeind
        pitchwens(i)=pitcheeind #in rad
    end
end	
longitgrd(i)=(.2*(pitch(i)-pitchwens(i))+.8*q(i))*180/pi #in deg
if longitgrd(i)>10
    longitgrd(i)=10
end
if longitgrd(i) < -10
    longitgrd(i)=-10
end
longit(i)=longitgrd(i)*pi/180	#in rad


#Defining the differential equations
#defining the nondimensional notations
qdiml(i)=q(i)/omega
vdiml(i)=sqrt(u(i)^2+w(i)^2)/vtip
if u(i)==0 
    if w(i)>0
        phi(i)=pi/2
    else
        phi(i)=-pi/2
    end
else
    phi(i)=atan(w(i)/u(i))
end
if u(i)<0
    phi(i)=phi(i)+pi
end
alfc(i)=longit(i)-phi(i)

mu(i)=vdiml(i)*cos(alfc(i))
labc(i)=vdiml(i)*sin(alfc(i))

#a1 calculi Flapping angle
teller(i)=-16/lok*qdiml(i)+8/3*mu(i)*collect(i)-2*mu(i)*(labc(i)+labi(i))
a1(i)=teller(i)/(1-.5*mu(i)^2)

# Thrust coeff from Bkade Element Method
ct_BEM(i)=cla*volh/4*(2/3*collect(i)*(1+1.5*mu(i)^2)-(labc(i)+labi(i)))
# Thrust coefficient from Glauert
alfd(i)=alfc(i)-a1(i)
ct_glau(i)=2*labi(i)*sqrt((vdiml(i)*cos(alfd(i)))^2+(vdiml(i)*...
sin(alfd(i))+labi(i))^2)

#Equations of motion
labidot(i)=ct_BEM(i) 
thrust(i)=labidot(i)*rho*vtip^2*area
helling(i)=longit(i)-a1(i)
vv(i)=vdiml(i)*vtip 		#it is 1/sqrt(u^2+w^2)

udot(i)=-g*sin(pitch(i))-cds/mass*.5*rho*u(i)*vv(i)+...
thrust(i)/mass*sin(helling(i))-q(i)*w(i)

wdot(i)=g*cos(pitch(i))-cds/mass*.5*rho*w(i)*vv(i)-...
thrust(i)/mass*cos(helling(i))+q(i)*u(i)

qdot(i)=-thrust(i)*mast/iy*sin(helling(i))

pitchdot(i)=q(i)

xdot(i)=u(i)*cos(pitch(i))+w(i)*sin(pitch(i))

zdot(i)=-c(i)

labidot(i)=(ct_BEM(i)-ct_glau(i))/tau
corrdot(i)=uwens-u(i)
corrcdot(i)=cwens(i)-c(i)

u(i+1)=u(i)+stap*udot(i)
w(i+1)=w(i)+stap*wdot(i)
q(i+1)=q(i)+stap*qdot(i)
pitch(i+1)=pitch(i)+stap*pitchdot(i)
x(i+1)=x(i)+stap*xdot(i)
labi(i+1)=labi(i)+stap*labidot(i)
z(i+1)=z(i)+stap*zdot(i)
corr(i+1)=corr(i)+stap*corrdot(i)
corrc(i+1)=corrc(i)+stap*corrcdot(i)
t(i+1)=t(i)+stap
end