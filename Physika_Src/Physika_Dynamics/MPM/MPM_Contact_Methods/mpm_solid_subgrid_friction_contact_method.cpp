/*
 * @file mpm_solid_subgrid_friction_contact_method.cpp 
 * @Brief an algorithm that can resolve contact between mpm solids with subgrid resolution,
 *        the contact can be no-slip/free-slip with Coulomb friction model
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cstdlib>
#include <cmath>
#include <limits>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/mpm_solid.h"
#include "Physika_Dynamics/MPM/MPM_Contact_Methods/mpm_solid_subgrid_friction_contact_method.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::MPMSolidSubgridFrictionContactMethod()
    :MPMSolidContactMethod<Scalar,Dim>(),
     friction_coefficient_(0.5),
     collide_threshold_(0.5),
     penalty_power_(6)
{
}

template <typename Scalar, int Dim>
MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::MPMSolidSubgridFrictionContactMethod(const MPMSolidSubgridFrictionContactMethod<Scalar,Dim> &contact_method)
    :friction_coefficient_(contact_method.friction_coefficient_),
     collide_threshold_(contact_method.collide_threshold_),
     penalty_power_(contact_method.penalty_power_)
{
    this->mpm_driver_ = contact_method.mpm_driver_;
}

template <typename Scalar, int Dim>
MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::~MPMSolidSubgridFrictionContactMethod()
{
}

template <typename Scalar, int Dim>
MPMSolidSubgridFrictionContactMethod<Scalar,Dim>& MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::operator=
(const MPMSolidSubgridFrictionContactMethod<Scalar,Dim> &contact_method)
{
    this->mpm_driver_ = contact_method.mpm_driver_;
    this->friction_coefficient_ = contact_method.friction_coefficient_;
    this->collide_threshold_ = contact_method.collide_threshold_;
    this->penalty_power_ = contact_method.penalty_power_;
    return *this;
}

template <typename Scalar, int Dim>
MPMSolidSubgridFrictionContactMethod<Scalar,Dim>* MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::clone() const
{
    return new MPMSolidSubgridFrictionContactMethod<Scalar,Dim>(*this);
}

template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::resolveContact(const std::vector<Vector<unsigned int,Dim> > &potential_collide_nodes,
                                                                      const std::vector<std::vector<unsigned int> > &objects_at_node,
                                                                      const std::vector<std::vector<Vector<Scalar,Dim> > > &normal_at_node,
                                                                      const std::vector<std::vector<unsigned char> > &is_dirichlet_at_node,
                                                                      Scalar dt)
{
    //for now, only contact between two objects in a cell is implemented
    resolveContactBetweenTwoObjects(potential_collide_nodes,objects_at_node,normal_at_node,is_dirichlet_at_node,dt);
}

template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::setFrictionCoefficient(Scalar coefficient)
{
    if(coefficient < 0)
    {
        std::cerr<<"Warning: invalid friction coefficient, 0.5 is used instead!\n";
        friction_coefficient_ = 0.5;
    }
    else
        friction_coefficient_ = coefficient;
}

template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::setCollideThreshold(Scalar threshold)
{
    if(threshold <= 0)
    {
        std::cerr<<"Warning: invalid collide threshold, 0.5 of the grid cell edge length is used instead!\n";
        collide_threshold_ = 0.5;
    }
    else if(threshold > 1)
    {
        std::cerr<<"Warning: collide threshold clamped to the cell size of grid!\n";
        collide_threshold_ = 1;
    }
    else
        collide_threshold_ = threshold;
}

template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::setPenaltyPower(Scalar penalty_power)
{
    if(penalty_power < 0)
    {
        std::cerr<<"Warning: invalid penalty, 6 is used instead!\n";
        penalty_power_ = 6;
    }
    else
        penalty_power_ = penalty_power;
}
    
template <typename Scalar, int Dim>
Scalar MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::frictionCoefficient() const
{
    return friction_coefficient_;
}

template <typename Scalar, int Dim>
Scalar MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::collideThreshold() const
{
    return collide_threshold_;
}

template <typename Scalar, int Dim>
Scalar MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::penaltyPower() const
{
    return penalty_power_;
}
    
template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::initParticleBucket(const std::set<unsigned int> &objects, ArrayND<std::map<unsigned int, std::vector<unsigned int> >,Dim> &bucket) const
{
    MPMSolid<Scalar,Dim> *mpm_solid_driver = dynamic_cast<MPMSolid<Scalar,Dim>*>(this->mpm_driver_);
    PHYSIKA_ASSERT(mpm_solid_driver);
    const Grid<Scalar,Dim> &grid = mpm_solid_driver->grid();
    Vector<unsigned int,Dim> grid_cell_num = grid.cellNum();
    bucket.resize(grid_cell_num);
    for(std::set<unsigned int>::iterator iter = objects.begin(); iter != objects.end(); ++iter)
    {
        unsigned int obj_idx = *iter;
        for(unsigned int particle_idx = 0; particle_idx < mpm_solid_driver->particleNumOfObject(obj_idx); ++particle_idx)
        {
            const SolidParticle<Scalar,Dim> &particle = mpm_solid_driver->particle(obj_idx,particle_idx);
            Vector<Scalar,Dim> particle_pos = particle.position();
            Vector<unsigned int,Dim>  bucket_idx;
            Vector<Scalar,Dim> bias_in_cell;
            grid.cellIndexAndBiasInCell(particle_pos,bucket_idx,bias_in_cell);
            std::map<unsigned int,std::vector<unsigned int> >::iterator map_iter = bucket(bucket_idx).find(obj_idx);
            if(map_iter == bucket(bucket_idx).end())
            {
                std::vector<unsigned int> vec(1,particle_idx);
                bucket(bucket_idx).insert(std::make_pair(obj_idx,vec));
            }
            else
                bucket(bucket_idx)[obj_idx].push_back(particle_idx);
        }
    }
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::tangentialDirection(const Vector<Scalar,Dim> &normal, const Vector<Scalar,Dim> &velocity_diff) const
{
    Vector<Scalar,Dim> tangent_dir = velocity_diff - velocity_diff.dot(normal)*normal;
    tangent_dir.normalize();
    return tangent_dir;
}

template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::adjacentCells(const Vector<unsigned int,Dim> &node_idx,
                                                                     const Vector<unsigned int,Dim> &cell_num,
                                                                     std::vector<Vector<unsigned int,Dim> > &cells) const
{
    cells.clear();
    Vector<unsigned int,Dim> cell_idx = node_idx;
    switch(Dim)
    {
    case 2:
    {
        for(unsigned int offset_x = 0; offset_x <= 1; ++offset_x)
            for(unsigned int offset_y = 0; offset_y <= 1; ++offset_y)
            {
                cell_idx[0] = node_idx[0] - offset_x;
                cell_idx[1] = node_idx[1] - offset_y;
                if(cell_idx[0] < 0 || cell_idx[1] < 0 || cell_idx[0] >= cell_num[0] || cell_idx[1] >= cell_num[1])
                    continue;
                cells.push_back(cell_idx);
            }
        break;
    }
    case 3:
    {
        for(unsigned int offset_x = 0; offset_x <= 1; ++offset_x)
            for(unsigned int offset_y = 0; offset_y <= 1; ++offset_y)
                for(unsigned int offset_z = 0; offset_z <= 1; ++offset_z)
            {
                cell_idx[0] = node_idx[0] - offset_x;
                cell_idx[1] = node_idx[1] - offset_y;
                cell_idx[2] = node_idx[2] - offset_z;
                if(cell_idx[0] < 0 || cell_idx[1] < 0 || cell_idx[2] < 0
                    ||cell_idx[0] >= cell_num[0] || cell_idx[1] >= cell_num[1] || cell_idx[2] >= cell_num[2])
                    continue;
                cells.push_back(cell_idx);
            }
        break;
    }
    default:
        PHYSIKA_ERROR("Wrong dimension specified!");
    }
}


template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::resolveContactBetweenTwoObjects(const std::vector<Vector<unsigned int,Dim> > &potential_collide_nodes,
                                                                                       const std::vector<std::vector<unsigned int> > &objects_at_node,
                                                                                       const std::vector<std::vector<Vector<Scalar,Dim> > > &normal_at_node,
                                                                                       const std::vector<std::vector<unsigned char> > &is_dirichlet_at_node,
                                                                                       Scalar dt)
{
    MPMSolid<Scalar,Dim> *mpm_solid_driver = dynamic_cast<MPMSolid<Scalar,Dim>*>(this->mpm_driver_);
    if(mpm_solid_driver == NULL)
    {
        std::cerr<<"Error: mpm driver and contact method mismatch, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    if(potential_collide_nodes.empty())  //no contact, direct return
        return;
    std::vector<std::vector<Vector<Scalar,Dim> > > normals = normal_at_node;  //normal will be modified to be colinear
    unsigned int collide_object_num = 2; //FOLLOWING METHOD IS ONLY CORRECT FOR TWO OBJECTS IN CONTACT!!!!
    //init the particle buckets of the involved objects
    std::set<unsigned int> involved_objects;
    for(unsigned int i = 0; i < objects_at_node.size(); ++i)
        for(unsigned int j = 0; j < collide_object_num; ++j)
            involved_objects.insert(objects_at_node[i][j]);
    ArrayND<std::map<unsigned int,std::vector<unsigned int> >,Dim> particle_bucket;
    initParticleBucket(involved_objects,particle_bucket);
    //resolve contact
    const Grid<Scalar,Dim> &grid = mpm_solid_driver->grid();
    Vector<unsigned int,Dim> grid_cell_num = grid.cellNum();
    for(unsigned int i = 0; i <potential_collide_nodes.size(); ++i)
    {
        Vector<unsigned int,Dim> node_idx = potential_collide_nodes[i];
        Vector<Scalar,Dim> node_pos = grid.node(node_idx);
        //first compute the center of mass velocity
        Vector<Scalar,Dim> vel_com(0);
        Scalar mass_com = 0;
        for(unsigned int j = 0; j < collide_object_num; ++j)
        {
            unsigned int obj_idx = objects_at_node[i][j];
            Scalar mass = mpm_solid_driver->gridMass(obj_idx,node_idx);
            Vector<Scalar,Dim> velocity = mpm_solid_driver->gridVelocity(obj_idx,node_idx);
            if(is_dirichlet_at_node[i][j])
            {
                //if any of the two colliding objects is dirichlet, then the center of mass velocity is the dirichlet velocity
                vel_com = velocity;
                mass_com = 1; //dummy mass
                break;
            }
            vel_com += mass*velocity;
            mass_com += mass;
        }
        vel_com /= mass_com;
        //average the normal of the two objects so that they're in opposite direction
        normals[i][0] = (normal_at_node[i][0] - normal_at_node[i][1]).normalize();
        normals[i][1] = -normals[i][0];
        //approximate the distance between objects with minimum distance between particles along the normal direction
        unsigned int obj_idx1 = objects_at_node[i][0], obj_idx2 = objects_at_node[i][1];
        std::vector<Vector<unsigned int,Dim> > adjacent_cells;
        adjacentCells(node_idx,grid_cell_num,adjacent_cells);
        std::vector<unsigned int> particles_obj1;
        std::vector<unsigned int> particles_obj2;
        for(unsigned int j = 0; j < adjacent_cells.size(); ++j)
        {
            Vector<unsigned int,Dim> cell_idx = adjacent_cells[j];
            std::vector<unsigned int> &particles_in_cell_obj1 = particle_bucket(cell_idx)[obj_idx1];
            std::vector<unsigned int> &particles_in_cell_obj2 = particle_bucket(cell_idx)[obj_idx2];
            particles_obj1.insert(particles_obj1.end(),particles_in_cell_obj1.begin(),particles_in_cell_obj1.end());
            particles_obj2.insert(particles_obj2.end(),particles_in_cell_obj2.begin(),particles_in_cell_obj2.end());
        }
        Scalar min_dist = (std::numeric_limits<Scalar>::max)();
        for(unsigned int j = 0; j < particles_obj1.size(); ++j)
            for(unsigned int k = 0; k < particles_obj2.size(); ++k)
            {
                const SolidParticle<Scalar,Dim> &particle1 = mpm_solid_driver->particle(obj_idx1,particles_obj1[j]);
                const SolidParticle<Scalar,Dim> &particle2 = mpm_solid_driver->particle(obj_idx2,particles_obj2[k]);
                Scalar dist = (particle1.position() - particle2.position()).dot(normals[i][0]);
                if(dist < 0)
                    dist = -dist;
                if(dist < min_dist)
                    min_dist = dist;
            }
        //resolve contact for each object at the node
        Vector<Scalar,Dim> trial_vel_obj1 = mpm_solid_driver->gridVelocity(obj_idx1,node_idx);
        Vector<Scalar,Dim> trial_vel_obj2 = mpm_solid_driver->gridVelocity(obj_idx2,node_idx);
        Vector<Scalar,Dim> vel_delta = trial_vel_obj1 - vel_com;
        Scalar vel_delta_dot_norm = vel_delta.dot(normals[i][0]);
        Scalar dist_threshold = collide_threshold_ * grid.minEdgeLength();
        if(vel_delta_dot_norm > 0 && min_dist < dist_threshold) //objects apporaching each other and close
        {
            //compute the tangential direction
            Vector<Scalar,Dim> tangent_dir = tangentialDirection(normals[i][0],vel_delta);
            //velocity difference in normal direction and tangential direction
            Scalar vel_delta_dot_tan = vel_delta.dot(tangent_dir);
            Vector<Scalar,Dim> vel_delta_norm = vel_delta_dot_norm * normals[i][0];
            Vector<Scalar,Dim> vel_delta_tan = vel_delta_dot_tan * tangent_dir;
            if(abs(vel_delta_dot_tan) > friction_coefficient_ * abs(vel_delta_dot_norm)) //slip with friction
                vel_delta_tan = friction_coefficient_ * vel_delta_norm;
            //apply a penalty function in the normal direction
            Scalar penalty_factor = 1 - pow(min_dist/dist_threshold,penalty_power_);
            vel_delta_norm *= penalty_factor;
            //update the grid velocity
            Vector<Scalar,Dim> new_vel_obj1 = trial_vel_obj1 - vel_delta_norm - vel_delta_tan;
            Vector<Scalar,Dim> new_vel_obj2 = trial_vel_obj2 + vel_delta_norm + vel_delta_tan;
            if(is_dirichlet_at_node[i][0] == 0x00)
                mpm_solid_driver->setGridVelocity(obj_idx1,node_idx,new_vel_obj1);
            if(is_dirichlet_at_node[i][1] == 0x00)
                mpm_solid_driver->setGridVelocity(obj_idx2,node_idx,new_vel_obj2);         
        }
    }
}

//explicit instantiations
template class MPMSolidSubgridFrictionContactMethod<float,2>;
template class MPMSolidSubgridFrictionContactMethod<float,3>;
template class MPMSolidSubgridFrictionContactMethod<double,2>;
template class MPMSolidSubgridFrictionContactMethod<double,3>;

}  //end of namespace Physika
