#include <iostream>
#include <fstream>
#include <math.h>
using namespace std;

/*
ifstream file("./feature.txt");
ofstream out("./feature_normalize.txt");

double features[2000][10];
double _min[10], _max[10];

int main() {
    for(int i = 0;i < 9;i++) _min[i] = 100000, _max[i] = 0;
    for (int i = 0;i < 1680; i++) {
        for (int j = 0;j < 9;j++) {
            file >> features[i][j];
            _min[j] = min(_min[j], features[i][j]);
            _max[j] = max(_max[j], features[i][j]);
        }
    }

    for(int i = 0;i < 1680; i++) {
        for(int j = 0;j < 9;j++) {
            features[i][j] = (features[i][j]-_min[j])/(_max[j]-_min[j]);
            if(j < 6) out << features[i][j];
            else out << features[i][j]*100;
            if(j != 8) out << ' ';
        }
        if(i != 1679) out << endl;
    }

}
*/

int main() {
    ifstream in("./out.txt");
    ofstream out1("./out1.txt");
    ofstream out2("./out2.txt");

    double a1, a2;
    while(in >> a1 >> a2) {
        out1 << a1 << endl;
        out2 << a2 << endl;
    }

}