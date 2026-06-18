// ============================================================
// C++17 reference implementation of the corridor pathfinding
// family, semantics-matched to experiments/run_all_experiments.py
// so that node-expansion counts are IDENTICAL (verify_parity.py).
//
// Algorithms: A*, ILS, AILS, JPS.
// Purpose: settle the "Python overhead" confound for a Q1 venue
//   - node counts are implementation-independent and must match Python
//   - wall-clock is measured here at compiled speed
//
// Instance format (stdin):
//   GRID H W
//   H lines of W chars: '.' free, '@' blocked
//   RISK 0|1
//   (if 1) H lines of W floats (space-separated)
//   QUERY sr sc gr gc lam
//   ALGO astar|ils|ails|jps
// Output (stdout, one line): algo,nodes,cost,time_ms,attempts,solved
// Build: make    Run: ./pathfind < instance.txt
// ============================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <queue>
#include <set>
#include <tuple>
#include <functional>
#include <limits>
#include <algorithm>
#include <chrono>
#include <string>
#include <iostream>
using namespace std;

static const double DIAG = 1.414;          // matches Python's 1.414 (not sqrt2)
static const double DIAGM1 = 1.414 - 1.0;

struct Grid {
    int H, W;
    vector<uint8_t> obs;     // 1 = blocked
    bool hasRisk = false;
    vector<double> risk;     // H*W
    inline bool free_(int r, int c) const {
        return r >= 0 && r < H && c >= 0 && c < W && !obs[(size_t)r * W + c];
    }
    inline double rk(int r, int c) const { return hasRisk ? risk[(size_t)r * W + c] : 0.0; }
};

static const int DR[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
static const int DC[8] = { 0, 0,-1, 1, -1,  1,-1, 1};
static const double DCOST[8] = {1,1,1,1,DIAG,DIAG,DIAG,DIAG};

inline double octile(int r1,int c1,int r2,int c2){
    int dr=abs(r1-r2), dc=abs(c1-c2);
    return (double)max(dr,dc) + DIAGM1 * (double)min(dr,dc);
}

// Open-list entry; tie-break by insertion counter to match Python heapq (f,counter)
struct Node { double f; long long counter; int r, c; };
struct NodeCmp {
    bool operator()(const Node&a, const Node&b) const {
        if (a.f != b.f) return a.f > b.f;        // min-heap on f
        return a.counter > b.counter;            // then FIFO insertion order
    }
};

inline size_t idx(int r,int c,int W){ return (size_t)r*W+c; }

// Generic corridor A*: mask==nullptr means unrestricted (plain A*).
// Returns nodes_expanded; fills path cost and solved flag.
long long corridorAStar(const Grid&g,int sr,int sc,int gr,int gc,double lam,
                        const vector<uint8_t>* mask, double& outCost, bool& solved,
                        vector<pair<int,int>>* outPath=nullptr){
    int H=g.H, W=g.W;
    vector<double> gscore((size_t)H*W, numeric_limits<double>::infinity());
    vector<int> came((size_t)H*W, -1);
    vector<uint8_t> closed((size_t)H*W, 0);
    priority_queue<Node,vector<Node>,NodeCmp> open;
    long long counter=0, nodes=0;
    gscore[idx(sr,sc,W)] = 0.0;
    open.push({octile(sr,sc,gr,gc), counter++, sr, sc});
    solved=false; outCost=0.0;
    while(!open.empty()){
        Node cur=open.top(); open.pop();
        int r=cur.r, c=cur.c; size_t ci=idx(r,c,W);
        if(closed[ci]) continue;
        closed[ci]=1;
        if(r==gr && c==gc){
            solved=true; outCost=gscore[ci];
            if(outPath){ outPath->clear(); int p=(int)ci;
                while(p!=-1){ outPath->push_back({p/W,p%W}); p=came[p]; }
                reverse(outPath->begin(), outPath->end()); }
            return nodes;
        }
        nodes++;
        for(int k=0;k<8;k++){
            int nr=r+DR[k], nc=c+DC[k];
            if(!g.free_(nr,nc)) continue;
            size_t ni=idx(nr,nc,W);
            if(mask && !(*mask)[ni]) continue;
            if(closed[ni]) continue;
            double risk_cost = (g.hasRisk && lam>0.0) ? lam*g.rk(nr,nc) : 0.0;
            double ng = gscore[ci] + DCOST[k] + risk_cost;
            if(ng < gscore[ni]){
                gscore[ni]=ng; came[ni]=(int)ci;
                open.push({ng+octile(nr,nc,gr,gc), counter++, nr, nc});
            }
        }
    }
    return nodes;
}

vector<pair<int,int>> bresenham(int r0,int c0,int r1,int c1){
    vector<pair<int,int>> cells; int dr=abs(r1-r0), dc=abs(c1-c0);
    int sr=r0<r1?1:-1, sc=c0<c1?1:-1, err=dr-dc, r=r0,c=c0;
    while(true){ cells.push_back({r,c}); if(r==r1&&c==c1) break;
        int e2=2*err; if(e2>-dc){err-=dc; r+=sr;} if(e2<dr){err+=dr; c+=sc;} }
    return cells;
}

void boxFill(vector<uint8_t>&mask,int H,int W,int lr,int lc,int rad){
    int r0=max(0,lr-rad), r1=min(H,lr+rad+1), c0=max(0,lc-rad), c1=min(W,lc+rad+1);
    for(int r=r0;r<r1;r++) for(int c=c0;c<c1;c++) mask[idx(r,c,W)]=1;
}

// ILS: incremental fixed-width corridor (matches ils_astar)
long long ils(const Grid&g,int sr,int sc,int gr,int gc,double lam,
              double initFrac,int maxAttempts,double&cost,bool&solved,int&attempts){
    int H=g.H,W=g.W;
    int diag=(int)sqrt((double)H*H+(double)W*W);
    int baseW=max(3,(int)(initFrac*diag));
    auto line=bresenham(sr,sc,gr,gc);
    long long total=0;
    for(int a=0;a<maxAttempts;a++){
        int cw=baseW + a*max(2,baseW/2);
        int hw=cw/2;
        vector<uint8_t> mask((size_t)H*W,0);
        for(auto&p:line) boxFill(mask,H,W,p.first,p.second,hw);
        double cc; bool sv; long long n=corridorAStar(g,sr,sc,gr,gc,lam,&mask,cc,sv);
        total+=n;
        if(sv){ cost=cc; solved=true; attempts=a+1; return total; }
    }
    solved=false; attempts=maxAttempts; return total;
}

// integral image (summed area) of obstacles, double, matches np.cumsum semantics
vector<double> integralObs(const Grid&g){
    int H=g.H,W=g.W; vector<double> I((size_t)H*W,0.0);
    for(int r=0;r<H;r++) for(int c=0;c<W;c++){
        double v=g.obs[idx(r,c,W)]?1.0:0.0;
        double up = r>0? I[idx(r-1,c,W)]:0.0;
        double lf = c>0? I[idx(r,c-1,W)]:0.0;
        double ul = (r>0&&c>0)? I[idx(r-1,c-1,W)]:0.0;
        I[idx(r,c,W)] = v + up + lf - ul;
    }
    return I;
}
double queryDensity(const vector<double>&I,int r,int c,int hw,int H,int W){
    int r0=max(0,r-hw), c0=max(0,c-hw), r1=min(H-1,r+hw), c1=min(W-1,c+hw);
    double A = I[idx(r1,c1,W)];
    double B = r0>0? I[idx(r0-1,c1,W)]:0.0;
    double C = c0>0? I[idx(r1,c0-1,W)]:0.0;
    double D = (r0>0&&c0>0)? I[idx(r0-1,c0-1,W)]:0.0;
    double s = A-B-C+D;
    double area = (double)(r1-r0+1)*(c1-c0+1);
    return area>0 ? s/area : 0.0;
}

// AILS: density-adaptive corridor (matches ails_astar)
long long ails(const Grid&g,int sr,int sc,int gr,int gc,double lam,
               int rmin,int rmax,double alpha,int omega,int maxAttempts,
               double&cost,bool&solved,int&attempts){
    int H=g.H,W=g.W;
    if(rmax<0) rmax=max(rmin+1,(int)(0.1*min(H,W)));
    auto I=integralObs(g);
    auto line=bresenham(sr,sc,gr,gc);
    long long total=0;
    for(int a=0;a<maxAttempts;a++){
        int bonus=a*max(1,rmin);
        vector<uint8_t> mask((size_t)H*W,0);
        for(auto&p:line){
            double dens=queryDensity(I,p.first,p.second,omega,H,W);
            int rad=(int)(rmin+(rmax-rmin)*pow(dens,alpha))+bonus;
            rad=max(rmin,min(rad,rmax+bonus));
            boxFill(mask,H,W,p.first,p.second,rad);
        }
        double cc; bool sv; long long n=corridorAStar(g,sr,sc,gr,gc,lam,&mask,cc,sv);
        total+=n;
        if(sv){ cost=cc; solved=true; attempts=a+1; return total; }
    }
    solved=false; attempts=maxAttempts; return total;
}

// -------- JPS (uniform cost, lam=0); node count = jump points expanded --------
struct Grid* GJ; // not used; keep simple closures via lambdas below

long long jps(const Grid&g,int sr,int sc,int gr,int gc,double&cost,bool&solved,
              vector<pair<int,int>>* outPath=nullptr){
    int H=g.H,W=g.W;
    auto free_=[&](int r,int c){ return g.free_(r,c); };
    function<pair<int,int>(int,int,int,int)> jump =
      [&](int r,int c,int dr,int dc)->pair<int,int>{
        int nr=r+dr, nc=c+dc;
        if(!free_(nr,nc)) return {-1,-1};
        if(nr==gr&&nc==gc) return {nr,nc};
        // forced neighbours
        if(dr!=0&&dc!=0){
            if((!free_(nr-dr,nc)&&free_(nr-dr,nc+dc)) ||
               (!free_(nr,nc-dc)&&free_(nr+dr,nc-dc))) return {nr,nc};
            if(jump(nr,nc,dr,0).first!=-1) return {nr,nc};
            if(jump(nr,nc,0,dc).first!=-1) return {nr,nc};
        } else if(dr!=0){
            if((!free_(nr,nc-1)&&free_(nr+dr,nc-1)) ||
               (!free_(nr,nc+1)&&free_(nr+dr,nc+1))) return {nr,nc};
        } else {
            if((!free_(nr-1,nc)&&free_(nr-1,nc+dc)) ||
               (!free_(nr+1,nc)&&free_(nr+1,nc+dc))) return {nr,nc};
        }
        return jump(nr,nc,dr,dc);
      };
    vector<double> gscore((size_t)H*W, numeric_limits<double>::infinity());
    vector<int> came((size_t)H*W,-1);
    vector<uint8_t> closed((size_t)H*W,0);
    priority_queue<Node,vector<Node>,NodeCmp> open;
    long long counter=0, nodes=0;
    gscore[idx(sr,sc,W)]=0.0; open.push({octile(sr,sc,gr,gc),counter++,sr,sc});
    solved=false; cost=0.0;
    while(!open.empty()){
        Node cur=open.top(); open.pop(); int r=cur.r,c=cur.c; size_t ci=idx(r,c,W);
        if(closed[ci]) continue; closed[ci]=1;
        if(r==gr&&c==gc){ solved=true; cost=gscore[ci];
            if(outPath){ outPath->clear(); int p=(int)ci; while(p!=-1){outPath->push_back({p/W,p%W}); p=came[p];} reverse(outPath->begin(),outPath->end()); }
            return nodes; }
        nodes++;
        for(int k=0;k<8;k++){
            auto jp=jump(r,c,DR[k],DC[k]);
            if(jp.first==-1) continue;
            int jr=jp.first, jc=jp.second; size_t ji=idx(jr,jc,W);
            if(closed[ji]) continue;
            double d=octile(r,c,jr,jc);
            double ng=gscore[ci]+d;
            if(ng<gscore[ji]){ gscore[ji]=ng; came[ji]=(int)ci;
                open.push({ng+octile(jr,jc,gr,gc),counter++,jr,jc}); }
        }
    }
    return nodes;
}

// -------------------- D* Lite (Koenig & Likhachev 2002) --------------------
// Initial plan from goal, then optional batch re-plan after blocking cells.
// Counts vertex expansions (U pops) across initial + replan; reports final cost.
struct DStarLite {
    const Grid& g; int H, W, S, T; double lam;
    vector<double> gv, rhs; double km = 0.0;
    // U as ordered set of (k1,k2,idx); keyOf holds current key for lazy remove
    set<tuple<double,double,int>> U;
    vector<pair<double,double>> keyOf; vector<uint8_t> inU;
    DStarLite(const Grid&g_,int s,int t,double lam_):g(g_),H(g_.H),W(g_.W),S(s),T(t),lam(lam_){
        gv.assign((size_t)H*W, INFINITY); rhs.assign((size_t)H*W, INFINITY);
        keyOf.assign((size_t)H*W, {0,0}); inU.assign((size_t)H*W,0);
    }
    inline double INFINITY_(){ return numeric_limits<double>::infinity(); }
    pair<double,double> calcKey(int u){
        double k2 = min(gv[u], rhs[u]);
        double k1 = (k2==INFINITY? INFINITY : k2 + octile(S/W,S%W,u/W,u%W) + km);
        return {k1,k2};
    }
    void uInsert(int u, pair<double,double> k){ keyOf[u]=k; inU[u]=1; U.insert({k.first,k.second,u}); }
    void uRemove(int u){ if(inU[u]){ auto k=keyOf[u]; U.erase({k.first,k.second,u}); inU[u]=0; } }
    double edge(int u,int v){ // cost u->v (entering v); INF if v blocked
        if(g.obs[v]) return INFINITY;
        int dr=abs(u/W - v/W), dc=abs(u%W - v%W);
        double d=(dr==1&&dc==1)?DIAG:1.0;
        return d + ((g.hasRisk&&lam>0)? lam*g.risk[v]:0.0);
    }
    void neighbors(int u, vector<int>&out){
        out.clear(); int r=u/W,c=u%W;
        for(int k=0;k<8;k++){ int nr=r+DR[k],nc=c+DC[k];
            if(nr<0||nr>=H||nc<0||nc>=W) continue; out.push_back(nr*W+nc); }
    }
    void updateVertex(int u){
        if(u!=T){ double m=INFINITY; vector<int> nb; neighbors(u,nb);
            for(int v:nb){ double c=edge(u,v); if(c<INFINITY && gv[v]<INFINITY) m=min(m,c+gv[v]); }
            rhs[u]=m; }
        uRemove(u);
        if(gv[u]!=rhs[u]) uInsert(u, calcKey(u));
    }
    long long computeShortestPath(){
        long long expansions=0;
        while(!U.empty()){
            auto top=*U.begin();
            pair<double,double> kold={get<0>(top),get<1>(top)};
            pair<double,double> kstart=calcKey(S);
            bool cont = (kold < kstart) || (rhs[S]!=gv[S]);
            if(!cont) break;
            int u=get<2>(top);
            pair<double,double> knew=calcKey(u);
            if(kold < knew){ uRemove(u); uInsert(u,knew); continue; }
            expansions++;
            if(gv[u] > rhs[u]){
                gv[u]=rhs[u]; uRemove(u);
                vector<int> nb; neighbors(u,nb); for(int v:nb) updateVertex(v);
            } else {
                gv[u]=INFINITY; uRemove(u);
                vector<int> nb; neighbors(u,nb); for(int v:nb) updateVertex(v);
                updateVertex(u);
            }
        }
        return expansions;
    }
    void init(){ rhs[T]=0.0; uInsert(T, calcKey(T)); }
    double pathCost(){ return gv[S]; } // cost from start to goal under current map
};

long long dstarLite(Grid&g,int sr,int sc,int gr,int gc,double lam,
                    const vector<pair<int,int>>&blocks, double&cost,bool&solved){
    int S=sr*g.W+sc, T=gr*g.W+gc;
    DStarLite d(g,S,T,lam); d.init();
    long long exp = d.computeShortestPath();
    // batch re-plan: block cells, update affected vertices, recompute
    if(!blocks.empty()){
        for(auto&b:blocks){ int v=b.first*g.W+b.second; if(!g.obs[v]){ g.obs[v]=1; } }
        vector<int> touched;
        for(auto&b:blocks){ int v=b.first*g.W+b.second; touched.push_back(v);
            int r=b.first,c=b.second; for(int k=0;k<8;k++){int nr=r+DR[k],nc=c+DC[k];
                if(nr>=0&&nr<g.H&&nc>=0&&nc<g.W) touched.push_back(nr*g.W+nc);} }
        for(int v:touched) d.updateVertex(v);
        exp += d.computeShortestPath();
    }
    cost = d.pathCost(); solved = (cost < numeric_limits<double>::infinity());
    return exp;
}

int main(){
    ios::sync_with_stdio(false);
    string tok; Grid g;
    if(!(cin>>tok)) return 1;            // GRID
    cin>>g.H>>g.W; g.obs.assign((size_t)g.H*g.W,0);
    for(int r=0;r<g.H;r++){ string row; cin>>row;
        for(int c=0;c<g.W;c++) g.obs[idx(r,c,g.W)] = (row[c]=='@')?1:0; }
    cin>>tok; int hr; cin>>hr;            // RISK 0|1
    if(hr){ g.hasRisk=true; g.risk.assign((size_t)g.H*g.W,0.0);
        for(int r=0;r<g.H;r++) for(int c=0;c<g.W;c++) cin>>g.risk[idx(r,c,g.W)]; }
    cin>>tok; int sr,sc,gr,gc; double lam; cin>>sr>>sc>>gr>>gc>>lam;  // QUERY
    cin>>tok; string algo; cin>>algo;     // ALGO
    // optional re-plan block for dstar:  REPLAN k  then k lines "r c"
    vector<pair<int,int>> blocks;
    {
        string maybe;
        if(cin>>maybe){
            if(maybe=="REPLAN"){ int kk; cin>>kk; for(int i=0;i<kk;i++){int br,bc; cin>>br>>bc; blocks.push_back({br,bc});} }
        }
    }
    double cost=0; bool solved=false; int attempts=1; long long nodes=0;
    auto t0=chrono::high_resolution_clock::now();
    if(algo=="astar"){ vector<uint8_t>* m=nullptr; nodes=corridorAStar(g,sr,sc,gr,gc,lam,m,cost,solved); }
    else if(algo=="ils"){ nodes=ils(g,sr,sc,gr,gc,lam,0.05,10,cost,solved,attempts); }
    else if(algo=="ails"){ nodes=ails(g,sr,sc,gr,gc,lam,2,-1,1.0,3,10,cost,solved,attempts); }
    else if(algo=="jps"){ nodes=jps(g,sr,sc,gr,gc,cost,solved); }
    else if(algo=="dstar"){ nodes=dstarLite(g,sr,sc,gr,gc,lam,blocks,cost,solved); }
    else { cerr<<"unknown algo\n"; return 2; }
    auto t1=chrono::high_resolution_clock::now();
    double ms=chrono::duration<double,milli>(t1-t0).count();
    printf("%s,%lld,%.10f,%.6f,%d,%d\n", algo.c_str(), nodes, cost, ms, attempts, solved?1:0);
    return 0;
}
