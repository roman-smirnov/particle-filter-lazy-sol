/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
namespace pfilter {

namespace {

constexpr int kNumParticles = 100;
constexpr double kDefaultWeight = 1.0;

constexpr double kTimeDelta = 0.1;  // Time elapsed between measurements [sec]
constexpr double kSensorRange = 50.0;  // Sensor range [m]

// GPS measurement uncertainty [x [m], y [m], theta [rad]]
constexpr double kGpsNoiseX = 0.3;
constexpr double kGpsNoiseY = 0.3;
constexpr double kGpsNoiseTheta = 0.01;

// Landmark measurement uncertainty [x [m], y [m]]
constexpr double kLandmarkNoiseX = 0.3;
constexpr double kLandmarkNoiseY = 0.3;

constexpr double kGaussNorm = 1.0 / (kTwoPi * kLandmarkNoiseX * kLandmarkNoiseY);

std::random_device true_random_engine; //true random in most implementations
std::default_random_engine pseudo_random_engine(true_random_engine());
std::normal_distribution<double> normal_distribution(0, 1);
}

const bool ParticleFilter::initialized() const {
  return !particles_list.empty();
}

void ParticleFilter::init(double x, double y, double theta) {
  for (int i = 0; i < kNumParticles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = x + normal_distribution(pseudo_random_engine) * kGpsNoiseX;
    particle.y = y + normal_distribution(pseudo_random_engine) * kGpsNoiseY;
    particle.theta = theta + normal_distribution(pseudo_random_engine) * kGpsNoiseTheta;
    particle.weight = kDefaultWeight;
    particles_list.push_back(particle);
  }
}

void ParticleFilter::prediction(double velocity, double yaw_rate) {
  yaw_rate = std::abs(yaw_rate) < kEpsilon ? 0 : yaw_rate;
  const double term = 0.5 * yaw_rate;
  const double delta_yaw = yaw_rate * kTimeDelta;
  const double coef = yaw_rate != 0 ? (2 * velocity / yaw_rate) * std::sin(0.5 * delta_yaw) : velocity * kTimeDelta;
  for (Particle& particle : particles_list) {
    particle.x += coef * std::cos(particle.theta + term) + normal_distribution(pseudo_random_engine) * kGpsNoiseX;
    particle.y += coef * std::sin(particle.theta + term) + normal_distribution(pseudo_random_engine) * kGpsNoiseY;
    particle.theta += delta_yaw + normal_distribution(pseudo_random_engine) * kGpsNoiseTheta;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  for (LandmarkObs& obs : observations) {
    double min_dist = kInfinity;
    LandmarkObs argmin_dist;
    for (LandmarkObs& prd : predicted) {
      double distance = dist(obs.x, obs.y, prd.x, prd.y);
      if (min_dist > distance) {
        min_dist = distance;
        argmin_dist = prd;
      }
    }
    obs.id = argmin_dist.id;
  }
}

// NOTE: The observations are given in the VEHICLE'S coordinate system.
// Your particles are located according to the MAP'S coordinate system
// Landmarks are also in MAP's coordinate system
void ParticleFilter::updateWeights(const std::vector<LandmarkObs>& observations, const Map& map_landmarks) {

  for (Particle& particle : particles_list) {

    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    // all landmarks within sensor range from particle
    std::vector<LandmarkObs> predictions;
    for (auto& landmark : map_landmarks.landmark_list) {
      double distance = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);
      if (distance <= kSensorRange) {
        predictions.push_back({landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }

    // transform observations from local vehicle coordinates (relative to particle) to global map coordinates
    std::vector<LandmarkObs> map_observations;
    for (const LandmarkObs& obs : observations) {
      double map_x = particle.x + obs.x * std::cos(particle.theta) - obs.y * std::sin(particle.theta);
      double map_y = particle.y + obs.x * std::sin(particle.theta) + obs.y * std::cos(particle.theta);
      map_observations.push_back({obs.id, map_x, map_y});
    }

    // associate (find and assign id) nearest landmark to each observation
    dataAssociation(predictions, map_observations);

    // reset particle weight
    particle.weight = kDefaultWeight;
    //find coordinates of nearest landmark and use to calculate weight
    for (const LandmarkObs& obs : map_observations) {
      for (const LandmarkObs& prd : predictions) {
        if (prd.id == obs.id) {
          const double x_exp = std::exp(-pow2(obs.x - prd.x) * 0.5 / pow2(kLandmarkNoiseX));
          const double y_exp = std::exp(-pow2(obs.y - prd.y) * 0.5 / pow2(kLandmarkNoiseY));
          particle.weight *= kGaussNorm * x_exp * y_exp;
          particle.associations.push_back(prd.id);
          particle.sense_x.push_back(prd.x);
          particle.sense_y.push_back(prd.y);
          break;
        }
      }
    }
  }
}

void ParticleFilter::resample() {
  // make vector of weights of all particles
  std::vector<double> weights(kNumParticles, kDefaultWeight);
  for (int i = 0; i < particles_list.size(); ++i) {
    weights[i] = particles_list[i].weight;
  }

  // weighted random index generator
  std::discrete_distribution<int> rand_ind_gen(weights.begin(), weights.end());

  // new particle population sampled from current particle population
  std::vector<Particle> new_particles(kNumParticles);
  for (Particle& new_p : new_particles) {
    const Particle& p = particles_list[rand_ind_gen(pseudo_random_engine)];
    new_p = {p.id, p.x, p.y, p.theta, p.weight, p.associations, p.sense_x, p.sense_y};
  }
  particles_list = new_particles;
}

std::string ParticleFilter::getAssociations(Particle best) {
  std::vector<int> v = best.associations;
  std::stringstream ss;
  std::copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseCoord(const Particle& best, std::string coord) {
  std::vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  std::copy(v.begin(), v.end(), std::ostream_iterator<double>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
//  std::cout << "best sense:" << v << std::endl;
//  for (auto b : v) {
//    std::cout << b << std::endl;
//  }
  return s;
}

}