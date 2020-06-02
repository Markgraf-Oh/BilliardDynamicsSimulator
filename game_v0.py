import numpy as np
import random

import pygame as pg
from pygame.locals import *
import sys

#물리엔진 관련
fps = 30
physical_accel = 6
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
size = 200.0 #table_len*size = 약 500
table_mid_x = edge + table_len*0.5*size
table_mid_y = edge + table_wid*0.5*size

#게임엔진 관련
pg.init()

screen = pg.display.set_mode((int(2.5*size + 2*edge), 600))
font = pg.font.SysFont('arial', 16)
clock = pg.time.Clock()
dodgerblue = pg.Color('dodgerblue2')
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
color_yellow = (125,125,0)
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
            if ball_table_u*g*(1/pps) >= abs(np.linalg.norm(point_v)):
                movingfriction = (-0.5)*point_v*pps
            else:
                movingfriction = A.mass*ball_table_u*g*(-1)*normal(point_v)
            A.a += movingfriction/A.mass
            A.alpha += (1/A.I)*np.cross(((-R)*z_hat), movingfriction)
        # 구름마찰력
        
        if rolling_u*g*(1/pps) <= np.linalg.norm(A.v):
            rollingfriction = (-1)*rolling_u * A.mass * g *normal(A.v)
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
    
def ball_draw():
    for a in range(0,len(ball.ball_list)):
        A = ball.ball_list[a]
        pg.draw.circle(screen,A.color,(int(table_mid_x + A.r[0]*size),int(table_mid_y - A.r[1]*size)),int(A.radius*size))
#객체 생성

west = wall("west",1.,0.,0.,(0.5*table_len))
east = wall("east",-1.,0.,0.,(0.5*table_len))
north = wall("north",0.,-1.,0.,(0.5*table_wid))
south = wall("south",0.,1.,0.,(0.5*table_wid))

#yellow1 = ball("yellow1",ball_mass,R)
#yellow2 = ball("yellow2",ball_mass,R)
#red1 = ball("red1",ball_mass,R)
white1 = ball("white1",ball_mass,R,color_white)

#객체 리스트
physics_list = ball.ball_list + wall.wall_list

#ball_locating()
white1.random_location(1)


#실행
while True:
    #이벤트(입력)을 받는다
    
    for event in pg.event.get():
        if event.type == pg.QUIT:
            #pg.quit()
            #sys.exit()
            break
        # 값을 입력중이지 않을 때, 입력을 선택하거나, 이미 입력된 값을 집어넣는다.
        if event.type == pg.KEYDOWN:
            if text_cursor == 0:
                if event.key == pg.K_LEFT:
                    text_cursor = 1
                    
                elif event.key == pg.K_RIGHT:
                    
                    text_cursor = 1
                    v_input = ''
                    wx_input = ''
                    wz_input =  ''
                    angle_input = ''
                    
                elif event.key == pg.K_DOWN:
                    #멈추고, 재위치
                    for obj in ball.ball_list:
                        obj.stop()
                    ball_locating()
                
                elif event.key == pg.K_SPACE:
                    #멈추기만
                    for obj in ball.ball_list:
                        obj.stop()
                    
                elif event.key == pg.K_RETURN:
                    # 입력중이지 않은 상태에서 엔터를 누르면 입력된 값이 대입된다.
                    # 하지만 입력값이 float으로 변환불가능하면 에러메세지 출력. 다시 입력해야함.
                    if working==0:
                        try:
                            white1.shoot(float(v_input), 2*np.pi*float(angle_input)/360, float(wx_input), float(wy_input), float(wz_input))
                            error = 0
                            working =1
                        except ValueError:
                            error = 1
                            print("wrong input")
        
            #속도입력
            if text_cursor == 1:
                if event.key == pg.K_RETURN:
                    text_cursor = 2
                elif event.key == pg.K_BACKSPACE:
                    v_input = v_input[:-1]
                else:
                    v_input += event.unicode
        
            #옆방향각속도입력
            elif text_cursor == 2:
                if event.key == pg.K_RETURN:
                    text_cursor = 3
                elif event.key == pg.K_BACKSPACE:
                    wx_input = wx_input[:-1]
                else:
                    wx_input += event.unicode

            #옆방향각속도입력
            elif text_cursor == 3:
                if event.key == pg.K_RETURN:
                    text_cursor = 4
                elif event.key == pg.K_BACKSPACE:
                    wy_input = wy_input[:-1]
                else:
                    wy_input += event.unicode

            #수직방향각속도입력
            elif text_cursor == 4:
                if event.key == pg.K_RETURN:
                    text_cursor = 5
                elif event.key == pg.K_BACKSPACE:
                    wz_input = wz_input[:-1]
                else:
                    wz_input += event.unicode
            #발사방향 입력
            elif text_cursor == 5:
                if event.key == pg.K_RETURN:
                    text_cursor = 0
                    working = 0
                elif event.key == pg.K_BACKSPACE:
                    angle_input = angle_input[:-1]
                else:
                    angle_input += event.unicode
    
    # 데이터 연산(물리엔진) 실시
    for tt in range(0,physical_accel):
        crashing()
        force()
        moving()
    
    # 기타 연산 : 출력문구 처리 등
    
    #에러문구
    if error == 0:
        errorcode = ''
    elif error == 1:
        errorcode = 'wrong input'
    
    
    # 화면갱신
    # 화면채우기
    screen.fill(color_white)
    
    # 입력란
    v_input_surface = font.render(('   V    (m/s) : ' + v_input), True, dodgerblue)
    wx_input_surface = font.render('Wx * R (m/s) : ' + wx_input, True, dodgerblue)
    wy_input_surface = font.render('Wy * R (m/s) : ' + wy_input, True, dodgerblue)
    wz_input_surface = font.render('Wz * R  (m/s) : ' + wz_input, True, dodgerblue)
    angle_input_surface = font.render(' angle  (`)   : ' + angle_input, True, dodgerblue)
    errorcode_surface = font.render(errorcode,True,dodgerblue)
    screen.blit(v_input_surface, (50, 400))
    screen.blit(wx_input_surface, (50, 430))
    screen.blit(wy_input_surface, (50, 460))
    screen.blit(wz_input_surface, (50, 490))
    screen.blit(angle_input_surface, (50, 520))
    screen.blit(errorcode_surface, (50, 550))
        
    # 상태란(속도절댓값, 각속도)
    v_track_surface = font.render(('   V    (m/s) : ' + str(abs(np.linalg.norm(white1.v)))), True, dodgerblue)
    w_track_surface = font.render(('   W*R  (m/s) : ' + str(abs(R*np.linalg.norm(white1.w)))), True, dodgerblue)
    wz_track_surface = font.render(('   Wz*R  (m/s) : ' + str(abs(R*np.inner(white1.w,z_hat)))), True, dodgerblue)

    screen.blit(v_track_surface, (250, 400))
    screen.blit(w_track_surface, (250, 430))
    screen.blit(wz_track_surface, (250, 460))
    
    #테이블
    pg.draw.rect(screen,color_green,(edge,edge,int(table_len*size),int(table_wid*size)))
    #pg.draw.circle(screen,color_white,(int(table_mid_x + white1.r[0]*size),int(table_mid_y - white1.r[1]*size)),int(R*size))
    ball_draw()
    
    #화면출력
    pg.display.flip()
    clock.tick(fps)
pg.quit()
sys.exit()