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
static std::normal_distribution<double> normal_distribution(0, 1);

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100;
  is_initialized = true;
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = x + normal_distribution(pseudo_random_engine) * std[0];
    particle.y = y + normal_distribution(pseudo_random_engine) * std[1];
    particle.theta = theta + normal_distribution(pseudo_random_engine) * std[2];
    particle.weight = 1;
    particles_list.push_back(particle);
    weights.push_back(1);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  yaw_rate = std::abs(yaw_rate) < 0.001 ? 0 : yaw_rate;
  const double term = 0.5 * yaw_rate;
  const double delta_yaw = yaw_rate * delta_t;
  const double coef = yaw_rate != 0 ? (2 * velocity / yaw_rate) * std::sin(0.5 * delta_yaw) : velocity * delta_t;
  for (Particle& particle : particles_list) {
    particle.x += coef * std::cos(particle.theta + term) + normal_distribution(pseudo_random_engine) * std_pos[0];
    particle.y += coef * std::sin(particle.theta + term) + normal_distribution(pseudo_random_engine) * std_pos[1];
    particle.theta += delta_yaw + normal_distribution(pseudo_random_engine) * std_pos[2];
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // find the particle closest to each observation
  for (LandmarkObs& obs : observations) {
    double min_dist = 999999;
    int argmin_dist_id = -1;
    for (LandmarkObs& prd : predicted) {
      double distance = dist(obs.x, obs.y, prd.x, prd.y);
      if (min_dist > distance) {
        min_dist = distance;
        argmin_dist_id = prd.id;
      }
    }
    obs.id = argmin_dist_id;
  }
}

// NOTE: The observations are given in the VEHICLE'S coordinate system.
// Your particles are located according to the MAP'S coordinate system
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {

  const double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

  for (Particle& particle : particles_list) {
    particle.weight = 1;
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    for (const LandmarkObs& obs : observations) {
      double pred_x = particle.x + obs.x * std::cos(particle.theta) - obs.y * std::sin(particle.theta);
      double pred_y = particle.y + obs.x * std::sin(particle.theta) + obs.y * std::cos(particle.theta);

      double min_dist = 999999;
      double nearest_x = 999999;
      double nearest_y = 999999;
      double nearest_id = 0;

      for (auto& landmark : map_landmarks.landmark_list) {
        double distance = dist(landmark.x_f, landmark.y_f, pred_x, pred_y);
        if (min_dist > distance) {
          min_dist = distance;
          nearest_x = landmark.x_f;
          nearest_y = landmark.y_f;
          nearest_id = landmark.id_i;
        }
      }
      const double x_exp = std::exp(-std::pow((obs.x - nearest_x), 2) / (2 * std::pow(std_landmark[0], 2)));
      const double y_exp = std::exp(-std::pow((obs.y - nearest_y), 2) / (2 * std::pow(std_landmark[1], 2)));
      particle.weight *= gauss_norm * x_exp * y_exp;

      particle.associations.push_back(nearest_id);
      particle.sense_x.push_back(nearest_x);
      particle.sense_y.push_back(nearest_y);
    }

    for (int i = 0; i < particles_list.size(); ++i) {
      weights[i] = particles_list[i].weight;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<Particle> new_particles_list(num_particles);
  for (int i = 0; i < particles_list.size(); ++i) {
    std::discrete_distribution<int> index(weights.begin(), weights.end());
    new_particles_list[i] = particles_list[index(pseudo_random_engine)];
  }
  particles_list = new_particles_list;

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