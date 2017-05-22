#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

#include <iostream>
using namespace std;

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
  cout << "---------------------" << endl;
  cout << "Predict Motion:\n" << endl;
  cout << "F_:\n" << F_ << endl;
  cout << "Q_:\n" << Q_ << endl;
  
  cout << "\nPa:\n" << P_ << endl;
  cout << "xa:\n" << x_ << endl;
  
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
  
  cout << "xb:\n" << x_ << endl;

  cout << "Pb:\n" << P_ << endl;

  cout << "---------------------" << endl;
}

void KalmanFilter::Update(const VectorXd &z) {
  cout << "---------------------" << endl;
  cout << "Update:\n" << endl;
  // Update the state by using standard Kalman Filter equations.
  
  // Intermediate calculations.
  MatrixXd Ht = H_.transpose();
  VectorXd y = z - H_ * x_;
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  long size = x_.size();
  MatrixXd I = MatrixXd::Identity(size, size);
  
  // Update state and covariance mats.
  cout << "H_:\n" << H_ << endl;
  cout << "y:\n" << y << endl;
  cout << "K:\n" << K << endl;
  
  cout << "\nPa:\n" << P_ << endl;
  cout << "xa:\n" << x_ << endl;
  
  x_ = x_ + K * y;
  P_ = (I - K * H_) * P_;
  cout << "xb:\n" << x_ << endl;
  cout << "Pb:\n" << P_ << endl;
  cout << "---------------------" << endl;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  cout << "---------------------" << endl;
  cout << "UpdateEKF:\n" << endl;
  // Update the state by using Extended Kalman Filter equations
  
  // Get predicted location in polar coords.
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  
  float eps = 0.000001;  // Make sure we don't divide by 0.
  if (px < eps && py < eps) {
    px = eps;
    py = eps;
  } else if (px < eps) {
    px = eps;
  }
  
  float rho = sqrtf(powf(px, 2) + powf(py, 2));
  float phi = atan2f(py, px);
  float rho_dot = (px * vx + py * vy) / rho;
  
  VectorXd hx(3);
  hx << rho, phi, rho_dot;
  
//  cout << "z:\n" << z << endl;
//  cout << "hx:\n" << hx << endl;
  
  // Intermediate calculations.
  MatrixXd Ht = H_.transpose();
  VectorXd y = z - hx;
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  long size = x_.size();
  MatrixXd I = MatrixXd::Identity(size, size);
  
  // Update state and covariance mats.
  cout << "H_:\n" << H_ << endl;
  cout << "y:\n" << y << endl;
  cout << "K:\n" << K << endl;
  
  cout << "\nPa:\n" << P_ << endl;
  cout << "xa:\n" << x_ << endl;
  
  x_ = x_ + K * y;
  P_ = (I - K * H_) * P_;
  cout << "xb:\n" << x_ << endl;
  cout << "Pb:\n" << P_ << endl;
  cout << "---------------------" << endl;
}
