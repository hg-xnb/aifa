#include <bits/stdc++.h>
#include <cstdio>

using namespace std;

// #define MODE 2
#define IO_FILE true


#define delta(a, b) (a)-(b)
#define squared(x) (x)*(x)

/// co-efficient
double a, b;

/// update-step
double alpha = 0.001;

/// Loss (MSE-method) value
double loss = 0;

/// Loss (MSE-method) values
vector<double> losses;

/// avergare accuracy sqrt(loss) value
double accuracy = 0;

/// dataset
struct point
{
    double x, y;
    point() { x = 0, y = 0; };
    template <class Tx, class Ty>
    point(Tx xp, Ty yp) { x = xp, y = yp; }
};
typedef vector<point> DS;
DS ds(100);

/// linear function
double y_hat(double x)
{
    return a + b * x;
}

/// read dataset from file
void read_dataset()
{

#if defined(IO_FILE) && IO_FILE == true
    freopen64("dataset.txt", "r", stdin);
    freopen64("log.txt", "w", stdout);
#endif

    for (point &p : ds)
    {
        cin >> p.x >> p.y;
    }
}

/// calc Root mean squared error
double calc_RMSE(){
    double res = 0.0;
    for (auto const &p : ds){
        res += squared( delta(y_hat(p.x), p.y) );
    }
    res /= double(ds.size());
    return sqrt(res);
}

///  update co-efficient
void update_w()
{
    double delta_y = 0.0;
    double delta_a = 0;
    double delta_b = 0;
    double rmse    = calc_RMSE(); 
    for (auto const &p : ds)
    {
        delta_y = delta(y_hat(p.x),p.y);
        delta_a +=       delta_y ;
        delta_b += p.x * delta_y ;
        /// cout << "y_hat - y = " << delta_y << '\n';
    }
    
    delta_a = delta_a / ( rmse * double(ds.size()) );
    delta_b = delta_b / ( rmse * double(ds.size()) );
    
    /// cout << "U: delta_a = " << delta_a << '\n';
    /// cout << "U: delta_b = " << delta_b << '\n';
    /// cout << "U: alpha * delta_a = " << alpha * delta_a << '\n';
    /// cout << "U: alpha * delta_b = " << alpha * delta_b << '\n';
    
    a -= alpha * delta_a;
    b -= alpha * delta_b;

    /// cout << "U: a = " << a << '\n';
    /// cout << "U: b = " << b << '\n';
}

/// calc loss using RMSE method
void calc_loss()
{
    loss = calc_RMSE();
}

/// calc accuracy
void calc_accuracy()
{
    accuracy = 0;
    for (auto const &p : ds)
    {
        
        if (p.y != 0)
            accuracy += abs(y_hat(p.x) - p.y)*100 / p.y;
        else
            accuracy += abs(y_hat(p.x) - p.y)*100;
    }
    accuracy /= double(ds.size());
}

int main()
{
    srand(time(NULL));

#if defined(MODE) && MODE == 0
    /// gen input
    freopen64("dataset.txt", "w", stdout);
    double n;
    for (int i = 1; i <= 700; i++){
        n = (rand()%137)*0.71;
        cout << n << "\t" << n - 0.15 * ((rand()%137)*0.71 - (rand()%37)*0.19) << '\n';
    }
    return 0;
#endif

#if !defined(MODE) || MODE > 0
    read_dataset();
    a =0, b = 0;
    for (int ep = 0; ep <= 26; ep++)
    {
        putchar('\n');
        printf("epoch [%d]:\n", ep);
        if(ep) update_w();
        calc_loss();
        calc_accuracy();
        losses.push_back(loss);
        cout << "a= " << a << '\n';
        cout << "b= " << b << '\n';
        cout << "loss= " << loss << '\n';
        cout << "avg error-rate= " << accuracy << "%\n";
    }

#if defined(MODE) && MODE == 2
    /// PRINT loss in process
    vector<vector<bool>> mat(100 + 4, vector<bool>(losses.size() + 4, false));
    for (int col = 0; col < losses.size(); ++col)
    {
        for (int row = 0; row < 100 * losses[col] / losses[0]; ++row)
        {
            mat[row][col] = 1;
        }
    }

    cout << "Loss in trainning process:\n";

    for (int row = 0; row < 100; ++row)
    {
        cout << 100 - row << '\t';
        for (int col = 0; col < losses.size(); ++col)
        {
            mat[99 - row][col] ? cout << '*' : cout << ' ';
        }
        cout << '\n';
    }
#endif

#if defined(MODE) && MODE == 3
    /// gen x-set and y-set
    cout << "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
    for(auto p:ds) cout << p.x << ',';
    cout << "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
    for(auto p:ds) cout << p.y << ',';
#endif

#endif
}