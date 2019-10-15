/**
 * particle_filter.h
 * 2D particle filter class.
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <string>
#include <vector>
#include "helper_functions.h"

namespace pfilter {

struct Particle {
  int id;
  double x;
  double y;
  double theta;
  double weight;
  std::vector<int> associations;
  std::vector<double> sense_x;
  std::vector<double> sense_y;
};

class ParticleFilter {
 public:
  // Constructor
  // @param num_particles Number of particles
  ParticleFilter() = default;

  // Destructor
  ~ParticleFilter() {}

  /**
   * init Initializes particle filter by initializing particles to Gaussian
   *   distribution around first position and all the weights to 1.
   * @param x Initial x position [m] (simulated estimate from GPS)
   * @param y Initial y position [m]
   * @param theta Initial orientation [rad]
   */
  void init(double x, double y, double theta);

  /**
   * prediction Predicts the state for the next time step using the process model.
   * @param velocity Velocity of car from t to t+1 [m/s]
   * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
   */
  void prediction(double velocity, double yaw_rate);

  /**
   * dataAssociation Finds which observations correspond to which landmarks 
   *   (likely by using a nearest-neighbors data association).
   * @param predicted Vector of predicted landmark observations
   * @param observations Vector of landmark observations
   */
  void dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations);

  /**
   * updateWeights Updates the weights for each particle based on the likelihood of the observed measurements.
   * @param observations Vector of landmark observations
   * @param map Map class containing map landmarks
   */
  void updateWeights(const std::vector<LandmarkObs>& observations,
                     const Map& map_landmarks);

  /**
   * resample Resamples from the updated set of particles to form the new set of particles.
   */
  void resample();

  /**
   * initialized Returns whether particle filter is initialized yet or not.
   */
  const bool initialized() const;

  /**
   * Used for obtaining debugging information related to particles.
   */
  std::string getAssociations(Particle best);
  std::string getSenseCoord(const Particle& best, std::string coord);

  // Set of current particles
  std::vector<Particle> particles_list;

};
}
#endif  // PARTICLE_FILTER_H_