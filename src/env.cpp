#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <list>
#include <vector>
#include <ctime>
using namespace glm; using namespace std;

// ------------------------ Engine & Constants ----------------------
const int ACTION_DIM = 10; // so khop co the dieu khien
const int STATE_DIM = 39;  // 11 xuong*3 + 6 thong so (hip pos/vel + 2 chan cham dat)
vec2 g(0.0f, -980.6f);
struct Engine {
    GLFWwindow* window;
    int WIDTH = 800, HEIGHT = 600;

    Engine () {
        if (!glfwInit()) {
            cerr << "Failed to initialize GLFW" << endl; exit(EXIT_FAILURE);
        }

        window = glfwCreateWindow(WIDTH, HEIGHT, "RL is pain", nullptr, nullptr);
        if (!window) {
            cerr << "Failed to create GLFW window" << endl;
            glfwTerminate(); exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(window);
        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        glViewport(0, 0, fbWidth, fbHeight);
    }
    void run() {
        glClear(GL_COLOR_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, (double)WIDTH, 0.0, (double)HEIGHT, -1.0, 1.0);
    }

    // ------ Helper Functions ------
    float cross(vec2 a, vec2 b) {
        return a.x*b.y - a.y*b.x;
    }
    void drawCircle(vec2 pos, float radius, vec3 color) {
        glColor3f(color.r, color.g, color.b);
        glBegin(GL_TRIANGLE_FAN);
            glVertex2f(pos.x, pos.y);
            for (int i = 0; i <= 20; i++) {
                float a = (i / 20.0f) * 2.0f * glm::pi<float>();
                glVertex2f(pos.x + cos(a)*radius, pos.y + sin(a)*radius);
            }
        glEnd();
    }

    // ------ Callbacks ------
    vec2 rotate(vec2 v, float a) {
        float c = cos(a), s = sin(a);
        return vec2(c*v.x - s*v.y, s*v.x + c*v.y);
    }
    vec2 cross(float s, vec2 v) {
        return vec2(-s*v.y, s*v.x);
    }
};
Engine engine;



// ------------------------ Bodies & Physics ------------------------
struct Bone {
    vec2 pos, vel;
    float angle, angVel;
    float halfLength, radius, mass, inertia, invMass, invInertia;
    vec2 force = vec2(0.0f);
    float torque = 0.0f;
    bool dragged = false;
    bool highlight = false; // bat khi xuong dang duoc dieu khien bang chuot

    Bone(vec2 p,float a,float l,float r,float m) : pos(p), vel(0), angle(a), angVel(0), halfLength(l), radius(r), mass(m) {
        float L=l*2, W=r*2;
        inertia=(m*(L*L+W*W))/12.f;
        invMass=1.f/m;
        invInertia=1.f/inertia;
    }
    vec2 worldPoint(vec2 local) const {
        float c=cos(angle), s=sin(angle);
        return pos + vec2(c*local.x - s*local.y, s*local.x + c*local.y);
    }
    void draw() {
        vec2 dir(cos(angle),sin(angle));
        vec2 p1=pos-dir*halfLength, p2=pos+dir*halfLength;

        if(glm::length(p2-p1)*glm::length(p2-p1)==0) return;

        vec2 d=glm::normalize(p2-p1), r=vec2(-d.y,d.x)*radius;
        vec2 c1=p1+d*radius, c2=p2-d*radius;

        // Draw Rectangle Portion
        if (highlight) glColor4f(1.0f, 0.55f, 0.1f, 1);
        else           glColor4f(1, 1, 1, 1);
        glBegin(GL_QUADS);
            glVertex2f((c2+r).x,(c2+r).y);
            glVertex2f((c2-r).x,(c2-r).y);
            glVertex2f((c1-r).x,(c1-r).y);
            glVertex2f((c1+r).x,(c1+r).y);
        glEnd();

        // Draw End Caps
        for(vec2 c:{c1,c2}){
            glBegin(GL_TRIANGLE_FAN);
            glVertex2f(c.x,c.y);
            for(int i=0;i<=20;i++){
                float a=i/20.f*2*glm::pi<float>();
                glVertex2f(c.x+cos(a)*radius,c.y+sin(a)*radius);
            }
            glEnd();
        }
    }
};
struct Joint {
    Bone* A, *B; // bones conected by this joint
    vec2 anchorA_local, anchorB_local; // local positions the joint connects to
    float maxTorque = 1e5f;

    // --- Controller Properties ---
    float targetAngle = 0;
    float stiffness = 1.0f;

    Joint(Bone* a, Bone* b, vec2 anchorA, vec2 anchorB, float targetAngle=0.0f, float stiffness=1.0f) : 
    A(a), B(b), anchorA_local(anchorA), anchorB_local(anchorB), targetAngle(targetAngle), stiffness(stiffness) { }

    void solve(float dt) {
        Bone* A = this->A; Bone* B = this->B;

        vec2 rA = engine.rotate(anchorA_local, A->angle), rB = engine.rotate(anchorB_local, B->angle);
        vec2 pA = A->pos + rA, pB = B->pos + rB;

        // position error
        vec2 C = pB - pA;
        // anchor velocities
        vec2 vA = A->vel + engine.cross(A->angVel, rA);
        vec2 vB = B->vel + engine.cross(B->angVel, rB);

        vec2 relVel = vB - vA;

        // --- Baumgarte stabilization ---
        float beta = 0.8f;
        vec2 bias = (beta / dt) * C;
        vec2 Cdot = relVel + bias;

        float k11 = A->invMass + B->invMass + A->invInertia*rA.y*rA.y + B->invInertia*rB.y*rB.y;
        float k12 = -A->invInertia*rA.x*rA.y - B->invInertia*rB.x*rB.y;
        float k21 = k12;
        float k22 = A->invMass + B->invMass + A->invInertia*rA.x*rA.x + B->invInertia*rB.x*rB.x;

        float det = k11*k22 - k12*k21;
        if(det == 0) return;

        float invDet = 1.0f / det;

        vec2 impulse;
        impulse.x = -( k22*Cdot.x - k12*Cdot.y) * invDet;
        impulse.y = -(-k21*Cdot.x + k11*Cdot.y) * invDet;

        A->vel -= impulse * A->invMass;
        B->vel += impulse * B->invMass;

        A->angVel -= engine.cross(rA, impulse) * A->invInertia;
        B->angVel += engine.cross(rB, impulse) * B->invInertia;
    }
    void applyTorque(float dt) {

        float angle   = B->angle - A->angle;
        float angVel  = B->angVel - A->angVel;
        float error   = targetAngle - angle;
        error -= 2.0f * glm::pi<float>() * floorf((error + glm::pi<float>()) / (2.0f * glm::pi<float>()));

        float k = stiffness * maxTorque;
        float d = 2.0f * sqrt(k * (A->inertia + B->inertia));  // critical damping

        float torque  = k * error - d * angVel;

        // clamp to maxTorque so it cant explode
        torque = glm::clamp(torque, -maxTorque, maxTorque);

        // convert to impulse scaled by dt
        float impulse = torque * dt;

        A->angVel -= impulse * A->invInertia;
        B->angVel += impulse * B->invInertia;
    }
};

struct Skeleton {
    vector<Bone*> bones;
    vector<Joint> joints;
    vec2 startPos;
    int idx;
    int impulse_timer = 0;
    int next_impulse_delay = 90;
    float impulse_max = 0.0f;

    Bone *head, *body, *armL, *armR, *forearmL, *forearmR, *hip, *legR, *legL, *calfR, *calfL;

    Skeleton(vec2 p, int idx) : startPos(p), idx(idx) { init(p); }

    void init(vec2 p, float jitter = 0.4f) {
        //srand(time(0) + idx * 1000);
        auto rnd = [&](){ return ((rand() % 1000) / 1000.0f - 0.5f) * 2.0f * jitter; };

        impulse_timer = 0;
        vec2 offset = p - vec2(400.0f, 60.0f);
        for(auto b : bones) delete b;
        joints.clear();

        // ---- Bones (mass = 1.0 dong nhat cho tat ca xuong - ragdoll dong nhat) ----
        head     = new Bone(vec2(400,200) + offset, 0.0f + rnd(),  12, 11,  1.0); 
        body     = new Bone(vec2(400,130) + offset, 1.50f + rnd(),  28, 15,  1.0); 
        hip      = new Bone(vec2(400,200) + offset, 6.36f + rnd(),  13, 13,  1.0); 

        armR     = new Bone(vec2(250,200) + offset, 4.45f + rnd(),  20,  7,  1.0); 
        armL     = new Bone(vec2(550,200) + offset, 5.48f + rnd(),  20,  7,  1.0);
        forearmR = new Bone(vec2(250,200) + offset, 4.45f + rnd(),  18,  6,  1.0);
        forearmL = new Bone(vec2(550,200) + offset, 5.48f + rnd(),  18,  6,  1.0);

        legR     = new Bone(vec2(250,200) + offset, 7.33f + rnd(),  28,  8,  1.0); 
        legL     = new Bone(vec2(550,200) + offset, 8.36f + rnd(),  28,  8,  1.0);
        calfR    = new Bone(vec2(250,200) + offset, 7.33f + rnd(),  24,  7,  1.0); 
        calfL    = new Bone(vec2(550,200) + offset, 8.36f + rnd(),  24,  7,  1.0);

        bones = {head,body,armL,armR,forearmL,forearmR,legL,legR,calfL,calfR,hip};

        // Joints with jittered target angles
        joints.push_back(Joint(body,head,{body->halfLength,0},{-head->halfLength,0}, 0.0f + rnd()));
        joints.push_back(Joint(body,hip,{-body->halfLength,0},{0,hip->halfLength}, 4.95f + rnd()));
        joints.push_back(Joint(body,armR,{body->halfLength*0.7f,-body->radius*0.94f},{armR->halfLength*0.95,0}, 3.0f + rnd()));
        joints.push_back(Joint(body,armL,{body->halfLength*0.7f, body->radius*0.94f},{armL->halfLength*0.95,0}, 4.0f + rnd()));
        joints.push_back(Joint(armR,forearmR,{-armR->halfLength,0},{forearmR->halfLength,0}, 0.0f + rnd()));
        joints.push_back(Joint(armL,forearmL,{-armL->halfLength,0},{forearmL->halfLength,0}, 0.0f + rnd()));
        joints.push_back(Joint(hip,legR,{-hip->radius*0.71f,-hip->radius*0.71f},{legR->halfLength,0}, 1.0f + rnd()));
        joints.push_back(Joint(hip,legL,{ hip->radius*0.71f,-hip->radius*0.71f},{legL->halfLength,0}, 2.0f + rnd()));
        joints.push_back(Joint(legR,calfR,{-legR->halfLength,0},{calfR->halfLength,0}, 0.0f + rnd()));
        joints.push_back(Joint(legL,calfL,{-legL->halfLength,0},{calfL->halfLength,0}, 0.0f + rnd()));

        // Initial constraint alignment
        for(auto& j : joints){
            vec2 err = j.A->worldPoint(j.anchorA_local) - j.B->worldPoint(j.anchorB_local);
            j.B->pos += err;
        }
    }

    void step(float dt) {
        // ---- Euler Integrate Gravity & Draw ----
        for (Bone* b : bones) {
            b->vel += g * dt;
            b->vel    *= 1.0f / (1.0f + 0.7f * dt);
            b->angVel *= 1.0f / (1.0f + 0.7f * dt);
            b->draw();
        }
        // ---- Solve Joints & Apply Torque---
        for(int i=0;i<50;i++) {
            for (Joint& j : joints){
                j.solve(dt);
                j.applyTorque(dt);
            }
        }
        // ---- Euler Integrate ----
        for (Bone* b : bones) {
            b->pos += b->vel * dt;
            b->angVel = glm::clamp(b->angVel, -50.0f, 50.0f);
            b->vel    = glm::clamp(b->vel, vec2(-6000.0f), vec2(6000.0f));
            b->angle += b->angVel * dt;
            checkBorderCollision(b);
        }

        // magic force — set to 0.0f for training, nonzero to test friction
        body->vel.x += 0.0f * dt;

        // foot friction — only calf bottom endpoints, only when grounded
        for (Bone* b : {calfL, calfR}) {
            vec2 foot = b->pos - vec2(cos(b->angle), sin(b->angle)) * b->halfLength;
            if (foot.y < b->radius + 2.0f)
                b->vel.x *= 0.05f;
        }
        ApplyImpulse();
    }
    void ApplyImpulse() {
        if (impulse_max <= 0.0f) return;
        impulse_timer++;
        if (impulse_timer < next_impulse_delay) return;
        impulse_timer = 0;
        next_impulse_delay = 60 + rand() % 121;

        Bone* targets[] = { body, head, armL, armR, legL, legR };
        float weights[] = { 0.5f, 0.2f, 0.1f, 0.1f, 0.05f, 0.05f };

        float r = (float)rand() / RAND_MAX;
        float cum = 0.0f;
        Bone* target = body;
        for (int i = 0; i < 6; i++) {
            cum += weights[i];
            if (r < cum) { target = targets[i]; break; }
        }

        float dir = (rand() % 2 == 0) ? 1.0f : -1.0f;
        float mag = (0.5f + 0.5f * ((float)rand() / RAND_MAX)) * impulse_max;
        // mag is calibrated as "velocity the body would get if
        target->vel.x += dir * mag * body->mass / target->mass;
    }

    void reset() {
        init(startPos, 0.3f);
    }
    void checkBorderCollision(Bone* b) {
        float restitution = 0.2f, slop = 0.01f, percent = 0.8f, friction = 0.99f;

        vec2 dir(cos(b->angle), sin(b->angle));
        vec2 offset = dir * b->halfLength;
        vec2 points[2] = { b->pos - offset, b->pos + offset };

        for (vec2 contact : points) {
            vec2 normal;
            float penetration = 0.0f;
            bool collided = false;

            // if (contact.x < b->radius) { normal = vec2(1,0); penetration =
            if (contact.y < b->radius) {
                normal = vec2(0,1);
                penetration = b->radius - contact.y;
                collided = true;
            }
            if (!collided) continue;

            // deal with the contacted point
            vec2 r = contact - b->pos;

            // velocity at contact
            vec2 vel = b->vel + engine.cross(b->angVel, r);

            float velAlongNormal = dot(vel, normal);
            if (velAlongNormal > 0) continue;

            float rCrossN = engine.cross(r, normal);
            float denom = b->invMass + (rCrossN * rCrossN) * b->invInertia;
            if (denom == 0) continue;

            float jn = -(1.0f + restitution) * velAlongNormal;
            jn /= denom;

            vec2 impulse = normal * jn;
            b->vel += impulse * b->invMass;
            b->angVel += engine.cross(r, impulse) * b->invInertia;


            // --- 3. Positional Correction ---
            vec2 correction = normal * percent * fmax(penetration - slop, 0.0f);
            b->pos += correction;
        }
    }
};
// ---- Single controllable skeleton (mo phong 1 khung nguoi duy nhat) ----
Skeleton* sk = new Skeleton(vec2(400, 60), 0);


// ------------------------ Interactive Mass ------------------------
struct CircleMass {
    vec2 pos=vec2(400,300), vel=vec2(0), dragOff=vec2(0);
    float mass=0, radius=20;
    bool active=false, dragged=false;
    bool cWas=false, vWas=false, upWas=false, dnWas=false, mbWas=false;

    void collide(Bone* b) {
        vec2 d(cos(b->angle),sin(b->angle));
        vec2 p1=b->pos-d*b->halfLength, p2=b->pos+d*b->halfLength, seg=p2-p1;
        float l2=dot(seg,seg);
        vec2 cp=p1+(l2>0?glm::clamp(dot(pos-p1,seg)/l2,0.f,1.f):0.f)*seg;
        vec2 delta=pos-cp; float dist=glm::length(delta), minD=radius+b->radius;
        if(dist>=minD||dist<1e-6f) return;
        vec2 n=delta/dist; float pen=minD-dist;
        vec2 r=cp+n*b->radius-b->pos;
        float rvn=dot(vel-(b->vel+engine.cross(b->angVel,r)),n);
        if(rvn>0) return;
        float invM=mass>0?1.f/mass:0.f, rcn=engine.cross(r,n);
        float j=-1.2f*rvn/(b->invMass+rcn*rcn*b->invInertia+invM);
        b->vel-=n*j*b->invMass; b->angVel-=engine.cross(r,n*j)*b->invInertia;
        if(mass>0) vel+=n*j*invM;
        float corr=std::max(pen-0.01f,0.f)*0.8f;
        float tot=b->invMass+invM;
        if(tot>0){ b->pos-=n*corr*b->invMass/tot; pos+=n*corr*(mass>0?invM/tot:1.f); }
    }

    void update(GLFWwindow* w, float dt) {
        bool c=glfwGetKey(w,GLFW_KEY_C)==GLFW_PRESS;
        if(c&&!cWas){double mx,my;glfwGetCursorPos(w,&mx,&my);pos=vec2(mx,engine.HEIGHT-my);vel=vec2(0);mass=0;active=true;}
        cWas=c;
        bool v=glfwGetKey(w,GLFW_KEY_V)==GLFW_PRESS;
        if(v&&!vWas) active=false; vWas=v;
        if(!active) return;

        bool up=glfwGetKey(w,GLFW_KEY_UP)==GLFW_PRESS;
        if(up&&!upWas){mass+=2;cout<<"mass="<<mass<<"kg\n";} upWas=up;
        bool dn=glfwGetKey(w,GLFW_KEY_DOWN)==GLFW_PRESS;
        if(dn&&!dnWas){mass=std::max(0.f,mass-2);cout<<"mass="<<mass<<"kg\n";} dnWas=dn;

        double mx,my; glfwGetCursorPos(w,&mx,&my);
        vec2 mouse(mx,engine.HEIGHT-my);
        bool mb=glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS;
        if(mb&&!mbWas&&glm::length(mouse-pos)<radius){dragged=true;dragOff=pos-mouse;}
        if(!mb) dragged=false; mbWas=mb;

        if(dragged){vel=(mouse+dragOff-pos)/dt;pos=mouse+dragOff;}
        else{
            if(mass>0) vel+=g*dt;
            vel*=1.f/(1.f+0.5f*dt);
            pos+=vel*dt;
            pos=glm::clamp(pos,vec2(radius),vec2(engine.WIDTH-radius,engine.HEIGHT-radius));
        }
        for(Bone* b:sk->bones) collide(b);
        engine.drawCircle(pos,radius,vec3(1,.5f,0));
    }
} circleMass;


// ------------------------ Mouse Control ------------------------
// Dieu khien truc tiep bang chuot (thay cho Python):
//   - Chuot TRAI gan mot khop (joint anchor) -> quay khop: targetAngle
//     cua khop theo huong con tro chuot so voi diem neo
//   - Chuot TRAI tren than xuong  -> keo xuong bang luc lo xo (drag)
//   - Chuot PHAI tren xuong       -> xoay truc tiep goc cua xuong
struct MouseController {
    enum Mode { NONE, ROTATE_JOINT, DRAG_BONE, HARD_ROTATE };
    Mode mode = NONE;
    Joint* activeJoint = nullptr;
    Bone*  activeBone  = nullptr;
    vec2 grabLocal = vec2(0); // diem grab trong he toa do local cua xuong
    bool lWas = false, rWas = false;

    vec2 mouseWorld(GLFWwindow* w) {
        double mx, my;
        glfwGetCursorPos(w, &mx, &my);
        return vec2((float)mx, engine.HEIGHT - (float)my);
    }

    // khoang cach tu diem p den "capsule" cua xuong, tra ve diem gan nhat
    float distToBone(Bone* b, vec2 p, vec2& closest) {
        vec2 dir(cos(b->angle), sin(b->angle));
        vec2 p1 = b->pos - dir * b->halfLength, p2 = b->pos + dir * b->halfLength;
        vec2 seg = p2 - p1;
        float l2 = dot(seg, seg);
        float t = l2 > 0 ? glm::clamp(dot(p - p1, seg) / l2, 0.0f, 1.0f) : 0.0f;
        closest = p1 + t * seg;
        return glm::length(p - closest);
    }

    // tim khop gan diem p tren xuong b (trong ban kinh 20px)
    Joint* jointNear(Bone* b, vec2 p) {
        Joint* best = nullptr; float bd = 20.0f;
        for (Joint& j : sk->joints) {
            if (j.A != b && j.B != b) continue;
            vec2 anchor = (j.A == b) ? j.A->worldPoint(j.anchorA_local)
                                     : j.B->worldPoint(j.anchorB_local);
            float d = glm::length(anchor - p);
            if (d < bd) { bd = d; best = &j; }
        }
        return best;
    }

    // tim xuong duoi con tro chuot (slack = sai so cho phep)
    Bone* boneAt(vec2 p, float slack, vec2& closestOut) {
        Bone* best = nullptr; float bd = slack;
        for (Bone* b : sk->bones) {
            vec2 cp;
            float d = distToBone(b, p, cp) - b->radius;
            if (d < bd) { bd = d; best = b; closestOut = cp; }
        }
        return best;
    }

    void clear() {
        mode = NONE; activeJoint = nullptr; activeBone = nullptr;
    }

    void update(GLFWwindow* w, float dt) {
        vec2 mouse = mouseWorld(w);
        bool l = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        bool r = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

        for (Bone* b : sk->bones) b->highlight = false;

        // ---- Grab khi vua nhan nut ----
        if (l && !lWas && !circleMass.active) {
            vec2 cp;
            Bone* b = boneAt(mouse, 30.0f, cp);
            if (b) {
                activeBone = b;
                Joint* j = jointNear(b, mouse);
                if (j) { mode = ROTATE_JOINT; activeJoint = j; }
                else   { mode = DRAG_BONE; grabLocal = engine.rotate(mouse - b->pos, -b->angle); }
            }
        }
        if (r && !rWas && !circleMass.active) {
            vec2 cp;
            Bone* b = boneAt(mouse, 30.0f, cp);
            if (b) { mode = HARD_ROTATE; activeBone = b; }
        }

        // ---- Ap dung trong khi giu nut ----
        if (l && mode == ROTATE_JOINT && activeJoint) {
            // xuong con (B) chi theo con tro chuot quanh diem neo khop
            vec2 pivot = activeJoint->A->worldPoint(activeJoint->anchorA_local);
            activeJoint->targetAngle =
                atan2(mouse.y - pivot.y, mouse.x - pivot.x) - activeJoint->A->angle;
            activeBone = activeJoint->B;
            activeBone->highlight = true;
        }
        else if (l && mode == DRAG_BONE && activeBone) {
            // keo xuong bang van toc dan ve phia con tro chuot
            Bone* b = activeBone;
            b->highlight = true;
            vec2 cur = b->worldPoint(grabLocal);
            vec2 targetVel = (mouse - cur) * 15.0f;
            float sp = glm::length(targetVel);
            if (sp > 4000.0f) targetVel *= 4000.0f / sp;
            b->vel = targetVel;
            b->angVel *= 0.8f;
        }
        else if (r && mode == HARD_ROTATE && activeBone) {
            // xoay truc tiep goc xuong theo huong chuot
            Bone* b = activeBone;
            b->highlight = true;
            vec2 d = mouse - b->pos;
            if (glm::length(d) > 5.0f) b->angle = atan2(d.y, d.x);
            b->angVel = 0.0f;
            b->vel *= 0.8f;
        }

        if (!l && !r) clear();
        lWas = l; rWas = r;

        // ---- Ve cac diem neo khop de de nhan dien ----
        for (Joint& j : sk->joints) {
            vec2 pivot = j.A->worldPoint(j.anchorA_local);
            bool hovered = glm::length(pivot - mouse) < 20.0f;
            bool active  = (&j == activeJoint && mode == ROTATE_JOINT);
            if (active)       engine.drawCircle(pivot, 7.0f, vec3(1.0f, 0.9f, 0.1f));
            else if (hovered) engine.drawCircle(pivot, 7.0f, vec3(0.2f, 1.0f, 0.3f));
            else              engine.drawCircle(pivot, 4.0f, vec3(0.55f, 0.55f, 0.6f));
        }
    }
} mouseCtl;
// ------------------------ UDP Connection (Python) ------------------------
// env.cpp nhan lenh/action tai port 5005, gui state den 127.0.0.1:5006
// Lenh tu Python (float dau tien cua goi tin):
//   -100.0            -> xin state (env tra loi bang sendData)
//   -150.0            -> yeu cau ve 1 frame (glfwSwapBuffers)
//   -69.0  + idx      -> reset skeleton
//   -68.0  + mag      -> dat impulse_max (luc day ngau nhien)
//   10 float          -> targetAngle cho 10 khop (thu tu joints)
// State gui di (39 float):
//   11 xuong * [sin(angle), cos(angle), angVel]  (thu tu bones)
//   + hip.pos.y/600, hip.pos.x/800, hip.vel.y/600, hip.vel.x/800
//   + calfL grounded, calfR grounded
struct Data {
    int sock, sendSock;
    sockaddr_in server, python;
    const static int revSize = ACTION_DIM;
    const static int sendSize = STATE_DIM;

    float recvBuffer[revSize], stateBuffer[sendSize];
    bool enabled = true; // P = bat/tat ket noi UDP
    bool pWas = false;

    Data() {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            cerr << "WSAStartup failed" << endl; exit(EXIT_FAILURE);
        }
        sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        sendSock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

        server.sin_family = AF_INET;
        server.sin_port = htons(5005);
        server.sin_addr.s_addr = INADDR_ANY;

        python.sin_family = AF_INET;
        python.sin_port = htons(5006);
        python.sin_addr.s_addr = inet_addr("127.0.0.1");

        bind(sock, (sockaddr*)&server, sizeof(server));

        // che do non-blocking: window khong treo khi Python chua ket noi
        u_long mode = 1;
        ioctlsocket(sock, FIONBIO, &mode);
    }

    bool receiveData(GLFWwindow* w) {
        // P = bat/tat UDP runtime
        bool p = glfwGetKey(w, GLFW_KEY_P) == GLFW_PRESS;
        if (p && !pWas) { enabled = !enabled; cout << "UDP " << (enabled ? "ON" : "OFF") << endl; }
        pWas = p;
        if (!enabled) return false;

        int bytesRead = recv(sock, (char*)recvBuffer, sizeof(recvBuffer), 0);

        if (bytesRead == SOCKET_ERROR) {
            int err = WSAGetLastError();
            if (err == WSAEWOULDBLOCK || err == WSAECONNRESET) return false;
        }

        if (bytesRead < (int)(2 * sizeof(float))) return false;

        if (recvBuffer[0] == -100.0f) {
            return true; // python xin state
        } else if (recvBuffer[0] == -150.0f) {
            glfwSwapBuffers(engine.window);
            return false;
        } else if (recvBuffer[0] == -69.0f) {
            sk->reset();
            mouseCtl.clear();
            return false;
        } else if (recvBuffer[0] == -68.0f) {
            sk->impulse_max = recvBuffer[1];
            return false;
        } else if (bytesRead == (int)(ACTION_DIM * sizeof(float))) {
            // action: 10 target angles -> gan vao cac khop
            for (int j = 0; j < ACTION_DIM && j < (int)sk->joints.size(); j++)
                sk->joints[j].targetAngle = recvBuffer[j];
        }
        return false;
    }

    void sendData() {
        int i = 0;
        for (Bone* b : sk->bones) {
            stateBuffer[i++] = sin(b->angle);
            stateBuffer[i++] = cos(b->angle);
            stateBuffer[i++] = b->angVel;
        }
        stateBuffer[i++] = sk->hip->pos.y / 600.0f;
        stateBuffer[i++] = sk->hip->pos.x / 800.0f;
        stateBuffer[i++] = sk->hip->vel.y / 600.0f;
        stateBuffer[i++] = sk->hip->vel.x / 800.0f;
        stateBuffer[i++] = (sk->calfL->pos.y - fabsf(sin(sk->calfL->angle)) * sk->calfL->halfLength <= sk->calfL->radius + 1.0f) ? 1.0f : 0.0f;
        stateBuffer[i++] = (sk->calfR->pos.y - fabsf(sin(sk->calfR->angle)) * sk->calfR->halfLength <= sk->calfR->radius + 1.0f) ? 1.0f : 0.0f;

        sendto(sendSock, (char*)stateBuffer, i * sizeof(float), 0, (sockaddr*)&python, sizeof(python));
    }
};
Data dataManager;


void tempKeyControl(GLFWwindow* w) {
    static int jointIdx = 0;
    static bool upPressed = false, downPressed = false, yPressed = false, rPressed = false;
    extern int frameInterval;

    // Y = doi toc do mo phong
    bool y = glfwGetKey(w, GLFW_KEY_Y) == GLFW_PRESS;
    if (y && !yPressed) { frameInterval = (frameInterval == 1) ? 8 : 1; cout << "frameInterval=" << frameInterval << endl; }
    yPressed = y;

    // R = reset skeleton
    bool rKey = glfwGetKey(w, GLFW_KEY_R) == GLFW_PRESS;
    if (rKey && !rPressed) {
        sk->reset();
        mouseCtl.clear(); // joints da duoc tao lai, con tro cu khong con hieu luc
        cout << "skeleton reset" << endl;
    }
    rPressed = rKey;

    float delta = 0.005f;
    int n = (int)sk->joints.size();

    // Up/Down = chon khop, Left/Right = tinh chinh target angle
    bool up = glfwGetKey(w, GLFW_KEY_UP) == GLFW_PRESS;
    if (up && !upPressed && !circleMass.active) {
        jointIdx = (jointIdx + 1) % n;
        cout << "Selected Joint Index: " << jointIdx << " target=" << sk->joints[jointIdx].targetAngle << endl;
    }
    upPressed = up;

    bool down = glfwGetKey(w, GLFW_KEY_DOWN) == GLFW_PRESS;
    if (down && !downPressed && !circleMass.active) {
        jointIdx = (jointIdx - 1 + n) % n;
        cout << "Selected Joint Index: " << jointIdx << " target=" << sk->joints[jointIdx].targetAngle << endl;
    }
    downPressed = down;

    if (glfwGetKey(w, GLFW_KEY_LEFT) == GLFW_PRESS)
        sk->joints[jointIdx].targetAngle -= delta;
    if (glfwGetKey(w, GLFW_KEY_RIGHT) == GLFW_PRESS)
        sk->joints[jointIdx].targetAngle += delta;

    // 1-9, 0 = tao impuse len xuong cua khop dang chon, huong ve con tro chuot
    static bool numPressed[10] = {};
    int numKeys[] = {GLFW_KEY_1,GLFW_KEY_2,GLFW_KEY_3,GLFW_KEY_4,GLFW_KEY_5,
                     GLFW_KEY_6,GLFW_KEY_7,GLFW_KEY_8,GLFW_KEY_9,GLFW_KEY_0};
    float mags[]  = {1500,2000,3000,4000,5000,6000,7000,8000,9000,10000};

    double cx, cy;
    glfwGetCursorPos(w, &cx, &cy);
    cy = engine.HEIGHT - cy;

    for (int k = 0; k < 10; k++) {
        bool pressed = glfwGetKey(w, numKeys[k]) == GLFW_PRESS;
        if (pressed && !numPressed[k]) {
            Bone* b = sk->joints[jointIdx].B;
            vec2 dir = vec2((float)cx, (float)cy) - b->pos;
            float len = glm::length(dir);
            if (len > 0) dir /= len;
            b->vel += dir * mags[k];
            cout << "Impulse " << mags[k] << " on joint " << jointIdx << endl;
        }
        numPressed[k] = pressed;
    }
}

// ------------------------ MAIN ------------------------
int timer = 0;
int frameInterval = 1; // vẽ mỗi frame - cần cho điều khiển chuột thời gian thực
int main() {
    srand(time(0) * 1234567891ULL ^ (uint64_t)clock());

    sk->impulse_max = 0.0f; // tat luc day ngau nhien (che do dieu khien thu cong)

    float dt = 1.0/60.0;
    glfwSwapInterval(1);
    glfwSwapBuffers(engine.window);
    while(!glfwWindowShouldClose(engine.window)) {
        engine.run();

        tempKeyControl(engine.window);

        // ------ DIEU KHIEN BANG CHUOT ------
        mouseCtl.update(engine.window, dt);

        // ------ NHAN LENH / ACTION TU PYTHON (UDP) ------
        bool gotStateRequest = dataManager.receiveData(engine.window);

        sk->step(dt);
        circleMass.update(engine.window, dt);

        // ------ GUI STATE CHO PYTHON (UDP) ------
        if (gotStateRequest)
            dataManager.sendData();

        timer++;
        if (timer % frameInterval == 0)
            glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    // Exit Program
    glfwTerminate(); return 0;
}