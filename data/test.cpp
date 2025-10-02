#include<bits/stdc++.h>
#define pb push_back
#define factorial(n) tgamma(n+1)
#define all(x) x.begin(),x.end()
#define yes cout<<"YES\n"
#define no cout<<"NO\n"
#define f(k,n) for(int i=k;i<n;i++)
#define test  int t; cin>>t; while(t--)
#define popcount(x) __builtin_popcountll(x);
using namespace std;
using ld = long double;
typedef long long ll;
typedef unsigned int ui;
typedef std::vector<int> vi;
typedef std::vector<ll> vl;
typedef std::vector<bool> vb;
typedef std::vector<char> vc;
const ll mod = 1000000007;
const int mod1 = 998244353;
ld phi = 3.14159;
string PHI="3.14159265358979323846264338327";
vector<ll>prm;

template<typename T> void in(T a,int n){f(0,n){cin>>a[i];}}
template<typename T> 
void out(T a,int n){f(0,n){cout<< a[i] <<"\n"[i==n-1];}}
template<typename T>
int s_in(T a,int n){int s=0;f(0,n){cin>>a[i];s+=a[i];}return s;}
template<typename T>
void erse(vector<T>& a,T b){a.erase(find(a.begin(),a.end(),b));}

bool ispow2(int x){ return x&&!(x&(x-1));}
ll gcd(ll a, ll b){return b==0?a:gcd(b,a%b);}
ll lcm(ll a,ll b){ return a/gcd(a,b)*b; }
int dg_sum(ll n){ int s; for(s=0;n>0;s+=n%10,n/=10); return s; }
ll binpow(ll a, ll b) {
    ll res = 1;
    while(b > 0) { 
        if(b&1) res *= a;
        a *= a; 
        b >>= 1;
    } 
    return res;
}

ll fib(ll n){ ld p=(1+sqrt(5))/2;
    return round(pow(p,n)/sqrt(5));
 //int a=0,b=1,c;if(n==0)return a;//f(2,n){c=a+b;a=b;b=c;}return b;
}
bool ispower(int &x,int &y){
	float res = log(y)/log(x);
	return res==(float)res;
}
void seo_prime(){
    vector<bool>p(10000000);
    prm.pb(2);
    ll i;
    for(i=3;i*i<10000001;i+=2){
        if(!p[i]){ prm.pb(i);
            for(int j=i*i;j<10000001;j+=i) p[j]=true;
        }
    }
    for(;i<10000000;i+=2) if(!p[i]) prm.pb(i);
        return;
}

int trail_zeroes(int n){
    int count = 0, i = 5;
    while(n/i > 0) {
        count += n / i;
        i *= 5;
    }
    return count;
}

int nCr(int n, int r){
    if(r > n) return 0;
    if(n == 0 || r == n) return 1;
    long long res = 1;
    for(int i = 1; i <= r; i++){
        res = res * (n - r + i) % MOD;
        res = res * binpow(i, MOD-2) % MOD;
    }
    return (int)res;
}

class DSU {
public:
    vector<int> parent, rank, size;
    DSU(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        size.resize(n, 1);
        iota(parent.begin(), parent.end(), 0);
    }
    int fup(int node) {
        if (node == parent[node]) return node;
        return parent[node] = fup(parent[node]);
    }
    void joins(int u, int v) {
        int p1 = fup(u);
        int p2 = fup(v);
        if (p1 == p2) return;
        if (size[p1] > size[p2]) size[p1] += size[p2], parent[p2] = p1;
        else size[p2] += size[p1], parent[p1] = p2;
    }
};

bool is_prime(ll n){
    if(n==2) return true;
    if(n<=1 || n%2==0) return false;
    for(int i=3;i*i<=n;i+=2){
        if(n%i==0) return false;
    }
    return true;
}

bool isPrime(ll n){
    if (n < 2) return 0;
    if (n == 2 || n == 3) return 1;
    if (n%2 == 0 || n%3 == 0) return 0;
    for (long long i=5; i*i<=n; i+=6) {
        if (n%i == 0 || n%(i+2) == 0)
            return 0;
    }
    return 1;
}

vector<int> mobius(int N) {
    vector<int> mu(N + 1, 1);
    vector<bool> isPrime(N + 1, true);

    for(int i = 2; i <= N; i++){
        if(isPrime[i]){
            for(int j = i; j <= N; j += i){
                isPrime[j] = false;
                mu[j] *= -1;
            }
            for(int j = i * i; j <= N; j += i * i) mu[j] = 0;
        }
    }
    return mu;
}

inline int read(){
    int x = 0, f = 1; char ch = getchar();
    while (!isdigit(ch)) { if (ch == '-') f = -1; ch = getchar();}
    while (isdigit(ch)) { x = x * 10 + ch - 48; ch = getchar();}
    return f == 1 ? x : -x;
}
inline void write(int x){
    if (x < 0) putchar('-'), x = -x;
    if (x > 9) write(x / 10);
    putchar(x % 10 + '0');
}
int main(){
    ios_base::sync_with_stdio(NULL); 
    cin.tie(NULL); cout.tie(NULL);
    seo_prime();
    test{
      int x = read();
      write(prm[x-1]);
      cout<<"hello";
    }
}
 
auto init = []() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    return 'c';
}();