#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
  0,      0.0225;
  
  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0,      0,
              0,    0.0009, 0,
              0,    0,      0.09;
  
  // Linear motion model.
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  
  // Init EKF matrices
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.F_ = MatrixXd(4, 4);
  
  // Init object covariance matrix.
  ekf_.P_ << 1, 0, 0,    0,
             0, 1, 0,    0,
             0, 0, 1000, 0,
             0, 0, 0,    1000;
  
  // Init state transition matrix. Indices (0, 2) and (1, 3) will be replaced
  // with dt at each measurement step.
  ekf_.F_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;
  
  // Don't need to init Q or H_j because they will be calculated from scratch
  // for every measurement.
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}


VectorXd PolarToCartesian(const VectorXd &polar_measurements) {
  float rho = polar_measurements(0);
  float phi = polar_measurements(1);
  float rho_dot = polar_measurements(2);
  
  float px = rho * cos(phi);
  float py = rho * sin(phi);
  float vx = rho_dot * cos(phi);
  float vy = rho_dot * sin(phi);
  
  VectorXd cartesian_measurements(4);
  cartesian_measurements << px, py, vx, vy;
  
  return cartesian_measurements;
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  long long timestamp = measurement_pack.timestamp_;
//  float timestamp = measurement_pack.timestamp_;
  if (!is_initialized_) {
    previous_timestamp_ = timestamp;
    
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
       Convert radar from polar to cartesian coordinates and initialize state.
       */
      //      float rho = measurement_pack.raw_measurements_(0);
      //      float phi = measurement_pack.raw_measurements_(1);
      //      float rho_dot = measurement_pack.raw_measurements_(2);
      
      VectorXd cartesian_measurements =
        PolarToCartesian(measurement_pack.raw_measurements_);
      
      float px = cartesian_measurements(0);
      float py = cartesian_measurements(1);
      float vx = cartesian_measurements(2);
      float vy = cartesian_measurements(3);
      
      ekf_.x_ << px, py, vx, vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
       Initialize state.
       */
      ekf_.x_ << measurement_pack.raw_measurements_(0),
                 measurement_pack.raw_measurements_(1),
                 0,
                 0;
    }
    
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // Get the time delta since the last measurement.
  float dt = (timestamp - previous_timestamp_) / 1000000.;  // Convert ms to s.
  previous_timestamp_ = timestamp;
  
//  cout << timestamp << endl;
//  cout << previous_timestamp_ << endl;
  
  // Update the state transition matrix F according to the new elapsed time.
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;
  
  // Update the process noise covariance matrix (Q).
  const float noise_ax = 9;
  const float noise_ay = 9;
  const float dt_2 = dt * dt;
  const float dt_3 = dt_2 * dt;
  const float dt_4 = dt_3 * dt;
  
  ekf_.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
             0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
             dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
             0, dt_3/2*noise_ay, 0, dt_2*noise_ay;
  
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    cout << "RADAR" << endl;
  } else {
    cout << "LASER" << endl;
  }
  ekf_.Predict();
  
  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }
}
