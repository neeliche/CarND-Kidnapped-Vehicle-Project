/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cfloat>

#include "particle_filter.h"

#define NUMBER_OF_PARTICLES 100


using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    num_particles = NUMBER_OF_PARTICLES;
    
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    default_random_engine gen;
    
    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        
        particles.push_back(p);
    }
    
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    //Create the normal distribution engine
    default_random_engine gen;
    
    //Initialize the noise for all three.
    normal_distribution<double> noise_x(0, std_pos[0]);
    normal_distribution<double> noise_y(0, std_pos[1]);
    normal_distribution<double> noise_theta(0, std_pos[2]);
    
    for(auto& particle: particles) {
        
        //Based on the yaw, the equation varies
        
        if(fabs(yaw_rate) < 0.00001) {
            particle.x += velocity * delta_t * cos(particle.theta);
            particle.y += velocity * delta_t * sin(particle.theta);
        } else {
            particle.x += (velocity/yaw_rate) * (sin(particle.theta + (yaw_rate * delta_t)) - sin(particle.theta));
            particle.y += (velocity/yaw_rate) * (cos(particle.theta) - cos(particle.theta + (yaw_rate * delta_t)));
            particle.theta += yaw_rate * delta_t;
        }
        
        particle.x += noise_x(gen);
        particle.y += noise_y(gen);
        particle.theta += noise_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (auto& obs : observations) {
        double minD = FLT_MAX;

        for (const auto& pred : predicted) {
            double distance = dist(obs.x, obs.y, pred.x, pred.y);
            if (minD > distance) {
                minD = distance;
                obs.id = pred.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    
    for(auto& particle: particles) {
        particle.weight = 1.0;
        
        //For each observation, convert to map coordinates
        std::vector<LandmarkObs> map_coord_observations;
        for(auto& observation: observations) {
            
            //X and Y observation 
            double conv_x = observation.x * cos(particle.theta) - observation.y * sin(particle.theta) + particle.x;
            double conv_y = observation.x * sin(particle.theta) + observation.y * cos(particle.theta) + particle.y;
            map_coord_observations.push_back(LandmarkObs{ observation.id, conv_x, conv_y});
        }

        //Find all valid nearby landmarks
        std::vector<LandmarkObs> predictions;
        for (auto& landmark : map_landmarks.landmark_list) {
            double distance = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);
            if (distance <= sensor_range) {
                predictions.push_back(LandmarkObs{ landmark.id_i, landmark.x_f, landmark.y_f  });
            }
        }
        
        //Associate observations with its nearest predictions
        dataAssociation(predictions, map_coord_observations);
        
        Map::single_landmark_s single_landmark;
        //Calculate the particle's weight
        //Done using the formula in 14.20
        for (auto& obs: map_coord_observations) {
            
            //Since I am not sure if the observation ids are ordered or not:
            for (const auto& landmark : map_landmarks.landmark_list) {
                if(landmark.id_i == obs.id) {
                    single_landmark = landmark;
                    break;
                }
            }
            
            double gausian_norm = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1]));
            double x_part = (pow((obs.x - single_landmark.x_f), 2)) / (2 * pow(std_landmark[0],2));
            double y_part = (pow((obs.y - single_landmark.y_f), 2)) / (2 * pow(std_landmark[1],2));
            double exponent = x_part + y_part;
            particle.weight *= (gausian_norm * exp(-exponent));
        }
        
        weights.push_back(particle.weight);
    }
    
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(weights.begin(), weights.end());
    
    vector<Particle> new_particles;
    
    for (int i=0; i<NUMBER_OF_PARTICLES; i++) {
        new_particles.push_back(particles[dist(gen)]);
    }
    
    particles = new_particles;
    weights.clear();

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
