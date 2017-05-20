#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

//#include <iostream>
//using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // Motion update.
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // Update the state by using standard Kalman Filter equations.
  
  // Intermediate calculations.
  MatrixXd Ht = H_.transpose();
  VectorXd y = z - H_ * x_;
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  long size = x_.size();
  MatrixXd I = MatrixXd::Identity(size, size);
  
  // Update state and covariance mats.
  x_ = x_ + K * y;
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Update the state by using Extended Kalman Filter equations
  
  // Get predicted location in polar coords.
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  
  float rho = sqrtf(powf(px, 2) + powf(py, 2));
  float phi = atan2f(py, px);
  float rho_dot = (px * vx + py * vy) / rho;
  
  VectorXd hx(3);
  hx << rho, phi, rho_dot;
  
  // Intermediate calculations.
  MatrixXd Ht = H_.transpose();
  VectorXd y = z - hx;
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  long size = x_.size();
  MatrixXd I = MatrixXd::Identity(size, size);
  
  // Update state and covariance mats.
  x_ = x_ + K * y;
  P_ = (I - K * H_) * P_;
}
