/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

static std::random_device true_random_engine; //true random in most implementations
static std::default_random_engine pseudo_random_engine(true_random_engine());
static std::normal_distribution<double> normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100;
  particles_list.reserve(num_particles);
  is_initialized = true;

  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = x + normal_distribution(pseudo_random_engine) * std[0];
    particle.y = y + normal_distribution(pseudo_random_engine) * std[1];
    particle.theta = theta + normal_distribution(pseudo_random_engine) * std[2];
    particle.weight = 1;
    particles_list.push_back(particle);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  const double term = 0.5 * yaw_rate;
  const double delta_yaw = yaw_rate * delta_t;
  const double coef = (yaw_rate != 0) ? (2 * velocity / yaw_rate) * std::sin(0.5 * delta_yaw) : velocity * delta_t;

  for (Particle& particle : particles_list) {
    particle.x += coef * std::cos(particle.theta + term) + normal_distribution(pseudo_random_engine) * std_pos[0];
    particle.y += coef * std::sin(particle.theta + term) + normal_distribution(pseudo_random_engine) * std_pos[1];
    particle.theta += delta_yaw + normal_distribution(pseudo_random_engine) * std_pos[2];
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (LandmarkObs& obs : observations) {

    double min_dist = 999999;
    for (LandmarkObs& prd : predicted) {
      double distance = dist(obs.x, obs.y, prd.x, prd.y);
      if(min_dist > distance ){
        min_dist = distance;
        obs.id = prd.id;
      }
    }

  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */


}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const std::vector<int>& associations,
                                     const std::vector<double>& sense_x,
                                     const std::vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best) {
  std::vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseCoord(Particle best, std::string coord) {
  std::vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}