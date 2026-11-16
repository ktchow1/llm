***************************
*** vec and noalias_vec ***
***************************
class vec 
{
public:
    double& operator[](int n)       { return ptr[n]; }
    double  operator[](int n) const { return ptr[n]; }
    int     size()            const { return N; }

public:
    // Being conservative : 
    // 1. materialize RHS into a temporary called "tmp"
    // 2. then copy tmp to *this
    // so that it is correct even if 'this' overlaps RHS (e.g. A = A + B + C) 

    template<class EXPRESSION>
    vec& operator=(const EXPRESSION& rhs) 
    {
        double* tmp = new double[N]; 
        for (int n=0; n!=N; ++n) tmp[n] = rhs[n]; // <--- materialize RHS expression here
        for (int n=0; n!=N; ++n) ptr[n] = tmp[n];
        delete[] tmp;
        return *this;
    }

    template<class EXPRESSION>
    vec& operator+=(const EXPRESSION& rhs) 
    {
        double* tmp = new double[N]; 
        for (int n=0; n!=N; ++n) tmp[n] = ptr[n] + rhs[n]; // <--- materialize RHS expression here
        for (int n=0; n!=N; ++n) ptr[n] = tmp[n];
        delete[] tmp;
        return *this;
    }

private:
    double* ptr;
    int     N;
};

class noalias_vec 
{
public:
    double& operator[](int n)       { return v[n]; }
    double  operator[](int n) const { return v[n]; }
    int     size()            const { return v.size(); }

public:
    template<class EXPRESSION>
    vec& operator=(const EXPRESSION& rhs) 
    {
        for (int n=0; n!=v.size(); ++n) v[n] = rhs[n]; // <-- in place
        return v;
    }

    template<class EXPRESSION>
    vec& operator+=(const EXPRESSION& rhs) 
    {
        for (int n=0; n!=v.size(); ++n) v[n] += rhs[n]; // <-- in place
        return v;
    }

private:
    vec& v;
};

// *** factory *** //
noalias_vec noalias(vec& v) 
{ 
    return noalias_vec{v}; 
}




******************************
*** Lazy expression of vec ***
******************************
class expression 
{    
public:          
    virtual ~expression() = default;
    virtual int    size()            const = 0;
    virtual double operator[](int n) const = 0; // <--- materialize here 
};

class add_expression : public expression // a + b
{ 
public:
    int    size()            const override { return a.size();    }
    double operator[](int n) const override { return a[n] + b[n]; }

private:
    const vec& a; 
    const vec& b;
};

struct scale_expression : public  expression // k * a
{  
public:
    int    size()            const override { return a.size(); }
    double operator[](int n) const override { return k * a[n]; }

private:
    double     k; 
    const vec& a;
};

struct smooth_expression : public expression 
{  
public:
    int    size()            const override { return a.size(); }
    double operator[](int n) const override 
    { 
        if      (n==0)        return  a[n];
        else if (n==size()-1) return  a[n];
        else                  return (a[n] + a[n-1] + a[n+1]) / 3.0; 
    }

private:
    const vec& a;
};

// *** factory *** //
   add_expression operator+(const vec& a, const vec& b) { return {a, b}; } // can be assigned to either vec or noalias_vec
 scale_expression operator*(const vec& a, double k)     { return {k, a}; } // can be assigned to either vec or noalias_vec
 scale_expression operator*(double k, const vec& a)     { return {k, a}; } // can be assigned to either vec or noalias_vec
smooth_expression    smooth(const vec& a)               { return {a};    } // can be assigned to vec only




// ************* //
// *** Usage *** //
// ************* //
vec a;
vec b;
        a  = a + b;     // ok, need alloc
        a  = k * a;     // ok, need alloc
        a  = smooth(a); // ok, need alloc (inevitable)
noalias(a) = a + b;     // ok, fast
noalias(a) = k * a;     // ok, fast
noalias(a) = smooth(a); // breaks

// Call sequence, for example : noalias(a) = a + b
//
// 1. operator+(a,b) ---> create add_expression    (RHS is a factory)
// 2. noalias(a)     ---> create noalias_vec       (LHS is a factory as well !!!)
// 3. noalias_vec::operator=(expression)            ... which runs a loop and invokes ...
//    (a) add_expression::operator[](int)
//    (b)            vec::operator[](int)           with vec a,b  inside add_expression::operator[](int)
//    (c)            vec::operator[](int)           with vec v,   inside    noalias_vec::operator=(expression) 

