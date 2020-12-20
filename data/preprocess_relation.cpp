#include <iostream>
#include <fstream>
#include <math.h>
#include <queue>
using namespace std;

ifstream file("./feature.txt");
ofstream outtrain("./train.txt");
ofstream outweight("./weight.txt");

double features[2000][10];
double sim[2000][2000];
const int ele_num = 1680;

double cosdis(double a[10], double b[10]) {
    double dis;
    double up = 0;
    double down1 = 0, down2 = 0;
    for(int i = 0;i < 9;i++) {
        up += a[i]*b[i];
        down1 += a[i]*a[i];
        down2 += b[i]*b[i];
    }
    return up/(sqrt(down1)+sqrt(down2));
}

double weight[2000][10];
int cnt[2000];

int main() {
    for (int i = 0;i < ele_num; i++) {
        for (int j = 0;j < 9;j++) {
            file >> features[i][j];
        }
    }

    for (int i = 0;i < ele_num; i++) {
        for(int j = 0;j < ele_num; j++) {
            sim[i][j] = cosdis(features[i], features[j]);
        }
    }

    for (int i = 0;i < ele_num; i++) {
        priority_queue<pair<double, int>> q ;
        for (int j = 0;j < ele_num;j++) {
            q.push({sim[i][j], j});
            if(q.size() > 10) q.pop();
        }

        while(!q.empty()) {
            auto cur = q.top();
            q.pop();
            for (int j = 0;j < 9;j++) {
                outtrain << features[i][j] << ' ';
            }
            for(int j = 0;j < 8;j++) {
                outtrain << features[cur.second][j] << ' ';
            }
            outtrain << features[cur.second][8] << endl;

            weight[i][cnt[i]++] = sim[i][cur.second];
        }
    }

    for(int i = 0;i < ele_num;i++) {
        double tot = 0;
        for(int j = 0;j < 10;j++) {
            tot += weight[i][j];
        }
        for(int j = 0;j < 10;j++) {
            weight[i][j] /= tot;
            if(j == 0) outweight << weight[i][j];
            else outweight << ' ' << weight[i][j];
        }
        outweight << endl;
    }

}