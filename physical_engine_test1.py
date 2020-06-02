import numpy as np
import random
import sys
import matplotlib.pyplot as plt

#물리엔진 관련
fps = 20
physical_accel = 9
pps = fps * physical_accel
table_len = 2.448
table_wid = 1.270
R = 0.03275  #당구공 반지름

ball_mass = 0.25 # 당구공 무게, kg


wall_wtrans = 0.1 # 벽과 공의 충돌 시 마찰 보정치
wall_e = 0.9 # 벽과 공의 탄성계수
wall_wtrans = 0.5 # 임시
ball_e = 0.98 # 공 사이의 탄성계수
rolling_u = 0.01 # 구름마찰계수
ball_table_u = 0.2 # 운동마찰계수
ball_ball_u = 0.05 # 공 사이의 운동마찰계수

ball_spin_dec = 10.0 #마찰에 의한 각속도 변화량 rad/s**2

g = 9.8 # 중력가속도 m/s**@

z_hat = np.array([0,0,1])
# 창크기 관련
edge = 50
size = 100.0 #table_len*size = 약 500
table_mid_x = edge + table_len*0.5*size
table_mid_y = edge + table_wid*0.5*size

#게임엔진 관련


v_input = ''
wx_input = ''
wy_input = ''
wz_input =  ''
angle_input = ''
text_cursor = 0
dummy = 0
error = 0
errorcode = ''
working = 0

#색지정
color_red = (255,0,0)
color_green = (0,255,0)
color_blue = (0,0,255)
color_yellow = (255,255,0)
color_white = (255,255,255)

class ball():
    ball_list = []
    def __init__(self, name, mass, radius, color):
        self.name = name
        self.mass = mass
        self.radius = radius
        self.I = 2*(mass*(radius**2))/5
        self.classnum = 1
        ball.ball_list.append(self)
        # ball.ball_list += [self]
        self.r = np.zeros(3,float)
        self.v = np.zeros(3,float)
        self.a = np.zeros(3,float)
        self.w = np.zeros(3,float)
        self.alpha = np.zeros(3,float)
        self.color = color
        
    def random_location(self,n):
        if n==0:
            self.r = np.array([random.uniform(R,(table_len/2)-R),random.uniform(R,(table_wid/2)-R),R])
        elif n==1:
            self.r = np.array([-random.uniform(R,(table_len/2)-R),random.uniform(R,(table_wid/2)-R),R])
        elif n==2:
            self.r = np.array([-random.uniform(R,(table_len/2)-R),-random.uniform(R,(table_wid/2)-R),R])
        elif n==3:
            self.r = np.array([random.uniform(R,(table_len/2)-R),-random.uniform(R,(table_wid/2)-R),R])
        else:
            self.r = np.array([random.uniform(-table_len/2 + R, table_len/2 - R),random.uniform(-table_wid/2 + R, table_wid/2 - R),R])
        
    def stop(self):
        self.v = np.zeros(3,float)
        self.a = np.zeros(3,float)
        self.w = np.zeros(3,float)
        self.alpha = np.zeros(3,float)
            
    def shoot(self, shoot_v, shoot_angle, shoot_wx, shoot_wy, shoot_wz):
        # wx와 wy는 vx와 vy를 기준으로 한다. [wx] x [wy] = - ( vx + vy)
        # 또한 그 단위는 각속도가 아닌, 구 표면의 속도를 m/s단위로 한다. 만약 wx
        if shoot_v == 0.0:
            shoot_v = 0.0001
        self.v = np.array([shoot_v*np.cos(shoot_angle),shoot_v*np.sin(shoot_angle),0.])
        self.w = np.array([0.,0.,shoot_wz/R])
        self.w += (shoot_wx/R)*normal(np.array([-shoot_v*np.sin(shoot_angle),shoot_v*np.cos(shoot_angle),0.0]))
        self.w += (shoot_wy/R)*normal(self.v)


class wall():
    wall_list = []
    def __init__(self, name, a, b, c, d):
        self.name = name
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.classnum = 2
        wall.wall_list += [self]
        self.N = np.array([self.a , self.b , self.c])/np.linalg.norm(np.array([self.a , self.b , self.c]))
    
    # 무한벽의 방정식은 wall.fuction() = 0
    def function(self, x, y, z):
        return ((self.a)*x + (self.b)*y + (self.c)*z + self.d)
    

    def distance(self, ra):
        dist = np.abs(self.function(ra[0],ra[1],ra[2])) / np.sqrt(self.a**2 + self.b**2 + self.c**2)
        return dist

# 함수정의

def normal(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def crash_detection(A,B):
    result=[0,0]
    if (A.classnum ==2) and (B.classnum==2):
        result = [2,2]
    elif (A.classnum ==1) and (B.classnum==1):
        if np.abs(np.linalg.norm(A.r - B.r)) <= (A.radius + B.radius):
            result = [1,1]
    elif (A.classnum ==1) and (B.classnum ==2):
        if B.distance(A.r) < A.radius:
            result = [1,2]
    elif (A.classnum == 2) and (B.classnum == 1):
        if A.distance(B.r) < B.radius:
            result = [2,1]
    else:
        result = [0,0]
    
    return result



def crashing():
    for a in range(0,len(physics_list)):
        for b in range(a+1,len(physics_list)):
            A = physics_list[a]
            B = physics_list[b]
            if crash_detection(A,B) == [1,2]:
                bf_w = np.copy(A.w)
                bf_v = np.copy(A.v)
                A.w += ((1+wall_e)/R)*((-1)*np.inner(bf_v , B.N))*np.cross(z_hat,B.N)
                A.v += (1+wall_e)*((-1)*np.inner(bf_v , B.N))*B.N
                # 마찰의 효과
                
                dv = (-1)*(np.sqrt(21.0/25.0)*R)*np.cross((np.inner(bf_w,z_hat)*z_hat),B.N) + (bf_v - np.inner(bf_v,B.N)*B.N)
                if abs(np.linalg.norm(dv)) > 0.0:
                    dp = A.mass*(1+wall_e)*((-1)*np.inner(bf_v , B.N))*ball_table_u*wall_wtrans
                    if dp >= abs(np.linalg.norm(dv)):
                        dp = 0.5*abs(np.linalg.norm(dv))
                    dp *= (-1)*normal(dv)
                    A.v += (dp/A.mass)
                    A.w += np.cross(((-1)*R*np.sqrt(21.0/25.0)*B.N), dp)/A.I
                A.v[2] = 0.0
                A.r += 0.5*A.v*(1/pps)
            elif crash_detection(A,B) == [2,1]:
                bf_w = np.copy(B.w)
                bf_v = np.copy(B.v)
                B.w += ((1+wall_e)/R)*((-1)*np.inner(bf_v , A.N))*np.cross(z_hat,A.N)
                B.v += (1+wall_e)*((-1)*np.inner(bf_v , A.N))*A.N
                # 마찰의 효과
                dv = (-1)*(np.sqrt(21.0/25.0)*R)*np.cross((np.inner(bf_w,z_hat)*z_hat),A.N) + (bf_v - np.inner(bf_v,A.N)*A.N)
                if abs(np.linalg.norm(dv)) > 0.0:
                    dp = B.mass*(1+wall_e)*((-1)*np.inner(bf_v , A.N))*ball_table_u*wall_wtrans
                    if dp >= abs(np.linalg.norm(dv)):
                        dp = 0.5*abs(np.linalg.norm(dv))
                    dp *= (-1)*normal(dv)
                    B.v += (dp/B.mass)
                    B.w += np.cross(((-1)*R*np.sqrt(21.0/25.0)*A.N), dp)/B.I
                B.v[2] = 0.0
                B.r += 0.5*B.v*(1/pps)
            elif crash_detection(A,B) == [1,1]:
                r_hat = normal(B.r - A.r)
                var = np.inner(A.v, r_hat)*r_hat
                vap = A.v - var
                wap = np.inner(A.w, np.cross(z_hat,r_hat))*np.cross(z_hat,r_hat)
                vbr = np.inner(B.v, r_hat)*r_hat
                vbp = B.v - vbr
                wbp = np.inner(B.w, np.cross(z_hat,r_hat))*np.cross(z_hat,r_hat)
                A.v = vap + ball_e*vbr
                A.w += (-1)*wap + ball_e*wbp
                B.v = vbp + ball_e*var
                B.w += (-1)*wbp + ball_e*wap
                A.r += 0.25*A.v*(1/pps)
                B.r += 0.25*B.v*(1/pps)

def force():
    # 가속도를 전부 초기화 한 뒤, 다시 부여한다.
    for a in range(0,len(ball.ball_list)):
        A = ball.ball_list[a]
        A.a = np.zeros(3,float)
        A.alpha = np.zeros(3,float)
    for a in range(0,len(ball.ball_list)):
        A = ball.ball_list[a]
        # 본래는 여기서 조건문으로 어떤 힘을 주어야 할지 정해야한다.
        # 그러나 본 시뮬에서는 모든 공이 테이블 위에서만 움직이기에 항상 같은 종류의 힘만 받으므로 생략
        # 운동마찰력
        point_v = A.v + (-R)*np.cross(A.w, z_hat)
        if abs(np.linalg.norm(point_v))>0.0:
            #if ball_table_u*g*(1/pps) >= abs(np.linalg.norm(point_v)):
            #    movingfriction = (-0.5)*point_v*pps
            #else:
            movingfriction = A.mass*ball_table_u*g*(-1)*normal(point_v)
            A.a += movingfriction/A.mass
            A.alpha += (1/A.I)*np.cross(((-R)*z_hat), movingfriction)
        # 구름마찰력
        if abs(np.linalg.norm(A.v)) != 0.0 :
            if rolling_u*g*(1/pps)*(7.0/abs(np.linalg.norm(A.v))) <= np.linalg.norm(A.v):
                rollingfriction = (-1)*rolling_u * A.mass * g *normal(A.v)*(7.0/abs(np.linalg.norm(A.v)))
                A.a += rollingfriction/A.mass
                A.alpha += np.cross(((-R)*z_hat), rollingfriction)/A.I
            else:
                rollingfriction = (-1)*A.mass*A.v*pps
                A.a += rollingfriction/A.mass
                A.alpha += np.cross(((-R)*z_hat), rollingfriction)/A.I
        
        # 마찰에 의한 z축 회전 감소
        if np.inner(A.w, z_hat) >= ball_spin_dec*(1/pps):
            A.alpha += (-1)*ball_spin_dec * normal(np.inner(A.w,z_hat)*z_hat)
        else:
            A.alpha += (-1)*np.inner(A.w,z_hat)*z_hat * pps

def moving():
    for a in range(0,len(ball.ball_list)):
        A = ball.ball_list[a]
        A.v[2] = 0.0
        A.r += (A.v * (1.0/pps)) + (0.5)*A.a*((1/pps)**2)
        A.v += A.a * (1/pps)
        A.v[2] = 0.0
        A.w += A.alpha * (1/pps)

def ball_locating():
    n = 0
    N = list(range(0,len(ball.ball_list)))
    random.shuffle(N)
    for obj in ball.ball_list:
        obj.random_location(N[n]%4)
        n+=1
    
#객체 생성

west = wall("west",1.,0.,0.,(0.5*table_len))
east = wall("east",-1.,0.,0.,(0.5*table_len))
north = wall("north",0.,-1.,0.,(0.5*table_wid))
south = wall("south",0.,1.,0.,(0.5*table_wid))

#yellow1 = ball("yellow1",ball_mass,R,color_yellow)
#yellow2 = ball("yellow2",ball_mass,R,color_yellow)
#red1 = ball("red1",ball_mass,R,color_red)
white1 = ball("white1",ball_mass,R,color_white)

#객체 리스트
physics_list = ball.ball_list + wall.wall_list

#ball_locating()
white1.r = np.array([-1.0,0.5,0.0])

#발사
v_input = 2.0
angle_input = 0
wx_input = -10.0
wy_input = 5.0
wz_input = 0.0
white1.shoot(v_input, 2*np.pi*angle_input/360, wx_input, wy_input, wz_input)
print(white1.v)
print(white1.w)

T = 300
X = []
Y = []

# 벽 그리기
x_max = int(table_len*5.0)#2.6*size 
y_max = int(table_wid*5.0)#1.3*size
plt.figure(figsize=(x_max,y_max),dpi=100)
plt.plot([0,0],[0,table_wid],'g-')#좌
plt.plot([table_len,table_len],[0,table_wid],'g-')#우
plt.plot([0,table_len],[table_wid,table_wid],'g-')#상
plt.plot([0,table_len],[0,0],'g-')#하

#엔진 시작
for time in range(0,T):
    for tt in range(0,physical_accel):
        crashing()
        force()
        moving()
    
    # 데이터 기록
    X.append(white1.r[0]+0.5*table_len)
    Y.append(white1.r[1]+0.5*table_wid)

plt.scatter(X,Y, c="r",marker='.')
plt.show()