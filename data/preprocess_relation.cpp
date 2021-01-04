#include <iostream>
#include <fstream>
#include <math.h>
#include <queue>
#include <string>
#include <cstring>
using namespace std;

string system_name = "BDBC";
ifstream file("./"+system_name+"_AllNumeric.txt");
ofstream outtrain("./"+system_name+"_train.txt");
ofstream outweight("./"+system_name+"_weight.txt");

const int max_size = 5000;
const int max_feature_size = 20;

double features[max_size][max_feature_size];
double sim[max_size][max_size];
const int ele_num = 2560;
const int train_size = 800;
const int neighbor_num = 10;
const int feature_num = 16;

double cosdis(double a[20], double b[20]) {
    double up = 0;
    double down1 = 0.01, down2 = 0.01;
    for(int i = 0;i < feature_num;i++) {
        up += a[i]*b[i];
        down1 += a[i]*a[i];
        down2 += b[i]*b[i];
    }
    return up/(sqrt(down1)+sqrt(down2));
}

double eucdis(double a[20], double b[20]) {
    double dis = 0;
    for(int i = 0;i < feature_num; i++) {
        dis += (a[i]-b[i])*(a[i]-b[i]);
    }
    dis += 0.00000001;
    return 1/sqrt(dis);
}

double weight[max_size][max_feature_size];
int cnt[max_size];

int main() {
    for (int i = 0;i < ele_num; i++) {
        for (int j = 0;j < feature_num;j++) {
            file >> features[i][j];
        }
    }

    for (int i = 0;i < ele_num; i++) {
        for(int j = 0;j < train_size; j++) {
            sim[i][j] = eucdis(features[i], features[j]);
        }
    }

    for (int i = 0;i < ele_num; i++) {
        priority_queue<pair<double, int>> q ;
        for (int j = 0;j < train_size;j++) {
            if(i == j) continue;
            q.push({sim[i][j], j});
            if(q.size() > neighbor_num) q.pop();
        }

        while(!q.empty()) {
            auto cur = q.top();
            q.pop();
            for (int j = 0;j < feature_num;j++) {
                outtrain << features[i][j] << ' ';
            }
            for(int j = 0;j < feature_num-1;j++) {
                outtrain << features[cur.second][j] << ' ';
            }
            outtrain << features[cur.second][feature_num-1] << endl;

            weight[i][cnt[i]++] = sim[i][cur.second];
        }
    }

    for(int i = 0;i < ele_num;i++) {
        double tot = 0;
        for(int j = 0;j < neighbor_num;j++) {
            tot += weight[i][j];
        }
        for(int j = 0;j < neighbor_num;j++) {
            weight[i][j] /= tot;
            if(j == 0) outweight << weight[i][j];
            else outweight << ' ' << weight[i][j];
        }
        outweight << endl;
    }

}