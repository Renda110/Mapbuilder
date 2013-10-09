#pragma once

#include <mrpt/base.h>
#include <mrpt/poses.h>
#include <mrpt/slam.h>
#include <mrpt/maps.h>
#include "/usr/local/include/mrpt/maps/include/mrpt/slam/COccupancyGridMap2D.h"
#include <map>
#include <cv.h>
#include <Eigen/Dense>

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/transform_broadcaster.h>

#include <boost/circular_buffer.hpp>
#include <boost/optional.hpp>
#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <boost/assign.hpp>

#include <cv.h>
#include <highgui.h>
#include <simplelog.h>
#include "robot_map_data_t.hpp"

#define DATA_PATH    "/home/enda/fuerte_workspace/sandbox/mapbuilder/data"
#define CONFIG_PATH  "/home/enda/fuerte_workspace/sandbox/mapbuilder/config"
#define PERSIST_FILE_HEADER "#MapBuilder v0.1"

#define VAR(V,init) __typeof(init) V=(init)
#define FOREACH(I,C) for(VAR(I,(C).begin());I!=(C).end();I++)

using namespace mrpt;
using namespace mrpt::poses;
using namespace mrpt::system;
using namespace mrpt::slam;
using namespace mrpt::utils;
using namespace std;
using namespace Eigen;

#define OCCGRID_CELLTYPE_MIN    (-128)
#define OCCGRID_CELLTYPE_MAX    (127)

#define D2R(x) (((x)/180.0)*3.141592653589)
#define R2D(x) (180.0*(x)/3.141592653589)
#define sqr(x) ((x)*(x))

#define WritePose( outfile, pose ) {                 \
	outfile.setf( ios::fixed, ios::floatfield );     \
	outfile.precision(3);                            \
	outfile << pose.x() << " " << pose.y() << " ";   \
	outfile.precision(6);                            \
	outfile << pose.phi() << " ";                    \
}

#define ReadPose( line, pose ) {     \
	double x,y,phi;                  \
	line >> x >> y >> phi;           \
	pose = CPose2D(x,y,phi);         \
}

#define WriteCov( outfile, cov ) {                                     \
	outfile.setf( ios::scientific, ios::floatfield );                  \
	outfile.precision(10);                                             \
	outfile << cov(0,0) << " " << cov(0,1) << " " << cov(0,2)  << " "; \
	outfile << cov(1,0) << " " << cov(1,1) << " " << cov(1,2)  << " "; \
	outfile << cov(2,0) << " " << cov(2,1) << " " << cov(2,2)  << " "; \
}

#define ReadCovariance( line, cov ) { for (int i=0; i<9; i++) line >> cov(i); }

//RR todo... yes this all needs refactoring

namespace mapping
{
	typedef string UUID;
	string CreateUUID();

	typedef enum ConstraintSourceType
	{
	    CONSTRAINT_TYPE_ODOMETRY,
	    CONSTRAINT_TYPE_SUBMAP_MATCHER,
	    CONSTRAINT_TYPE_HMI,
	    CONSTRAINT_TYPE_FIDUCIARY
	} ConstraintSourceType;

	//these are sent as DDS messages, treat them as immutable objects.
    class SubmapPose
    {
		UUID submap_uuid;
		CPose2D mean;
		CMatrixDouble33 covariance;
		CMatrixDouble33 precision;

        public:
        bool fixed; //is this submap's pose optimisable

        SubmapPose( UUID _submap_uuid, const CPose2D &_mean, const CMatrixDouble33 &_covariance, bool _fixed ) :
        submap_uuid(_submap_uuid),
            mean(_mean),
            covariance(_covariance),
            precision(_covariance.inv()),
            fixed(_fixed)
        {
        }

        SubmapPose( UUID _submap_uuid, const CPosePDFGaussian &pose, bool _fixed ) :
        submap_uuid(_submap_uuid),
            mean(pose.mean),
            covariance(pose.cov),
            precision(pose.cov.inv()),
            fixed(_fixed)
        {
        }

        SubmapPose( stringstream &line )
        {
            line >> submap_uuid;
            line >> fixed;
            ReadPose(line, mean);
            ReadCovariance(line, covariance);
            precision = covariance.inv();
        }

        void WriteToAsciiStream( ofstream &outfile )
        {
            outfile << "SubmapPose " << submap_uuid.c_str() << " ";
            outfile << fixed << " ";
            WritePose( outfile, mean );
            WriteCov(  outfile, covariance );
            outfile  << endl;
        }

        inline const UUID& GetUUID()
        {
            return submap_uuid;
        }

        inline const CPose2D& GetMean()
        {
            return mean;
        }

        inline const CMatrixDouble33& GetCovariance()
        {
            return covariance;
        }

        inline const CMatrixDouble33& GetPrecision()
        {
            return precision;
        }

        CPosePDFGaussian GetPosePDFGaussian()
        {
            return CPosePDFGaussian(mean,covariance);
        }

        //these were meant to be immutable and not have setters
        //however local-only optimisation and DDS issues have changed that
        inline void SetMean( CPose2D &_mean )
        {
            mean = _mean;
        }
	};

    class SubmapGroundTruth
    {
        public:
        UUID submap_uuid;
        CPose2D mean;
        Matrix3d covariance;
        Matrix3d precision;
        int optimisation_id;

        SubmapGroundTruth( UUID _submap_uuid, const CPose2D &_mean, const CMatrixDouble33 &_covariance ) :
        submap_uuid(_submap_uuid), mean(_mean)
        {
            for (int i = 0;i < 9;i++)
            {
                covariance(i) = _covariance(i);
            }

            precision = covariance.inverse();
            precision(0,1) = precision(1,0);
            precision(0,2) = precision(2,0);
            precision(1,2) = precision(2,1); //correct for minute numerical errors
        }

        SubmapGroundTruth( stringstream &line )
        {
            line >> submap_uuid;

            ReadPose(line, mean);
            ReadCovariance(line, covariance);

            precision = covariance.inverse();
            precision(0,1) = precision(1,0);
            precision(0,2) = precision(2,0);
            precision(1,2) = precision(2,1); //correct for minute numerical errors
        }

        void WriteToAsciiStream( ofstream &outfile )
        {
            outfile << "SubmapGroundTruth " << submap_uuid.c_str() << " ";
            WritePose( outfile, mean );
            WriteCov(  outfile, covariance );
            outfile  << endl;
        }

        inline const UUID& GetSubmapUUID()
        {
            return submap_uuid;
        }

        inline const CPose2D& GetMean()
        {
            return mean;
        }

        /*
        inline CMatrixDouble33 GetCovariance()
        {
            CMatrixDouble33 cov;
            for (int i=0;i<9;i++) cov.(i) = covariance(i);
            return cov;
        }
        */
	};

	//forward declare:
	class Submap;

    class SubmapConstraint
    {
        public:
		UUID uuid;
		UUID base_submap_uuid;
		UUID target_submap_uuid;
		Submap *base_submap; //set when added to Map
		Submap *target_submap;//set when added to Map
		CPose2D constraint_mean;
		Matrix3d constraint_covariance;
		ConstraintSourceType  constraint_type;
		double confidence;
		Matrix3d constraint_precision;

		SubmapConstraint( UUID _uuid, UUID _base_submap_uuid, UUID _target_submap_uuid, const CPose2D &_mean, const Matrix3d &_covariance, ConstraintSourceType _constraint_type, double _confidence ) :
		uuid( _uuid ),
			base_submap_uuid( _base_submap_uuid ),
			target_submap_uuid( _target_submap_uuid ),
			constraint_mean( _mean ),
			constraint_covariance( _covariance ),
			constraint_type(_constraint_type),
			confidence(_confidence)
		{
			//for stability, make sure the prec matrix is definitely positive semi-definite.
			//the inverse may have corrupted the floats slightly
			constraint_precision = _covariance.inverse();
			constraint_precision(0,1) = constraint_precision(1,0);
			constraint_precision(0,2) = constraint_precision(2,0);
			constraint_precision(1,2) = constraint_precision(2,1);
		}

        SubmapConstraint( stringstream &line )
        {
			line >> uuid;
			line >> base_submap_uuid;
			line >> target_submap_uuid;

			int i;
			line >> i;
			constraint_type = (ConstraintSourceType)i;
			line >> confidence;

			ReadPose(line, constraint_mean);
            ReadCovariance(line, constraint_covariance)
                    ;
			constraint_precision = constraint_covariance.inverse();
			constraint_precision(0,1) = constraint_precision(1,0);
			constraint_precision(0,2) = constraint_precision(2,0);
			constraint_precision(1,2) = constraint_precision(2,1);
		}

        void WriteToAsciiStream( ofstream &outfile )
        {
			outfile << "SubmapConstraint " << uuid.c_str() << " " << base_submap_uuid.c_str() << " " << target_submap_uuid.c_str() << " ";
			outfile << (int)constraint_type << " ";
			outfile.setf( ios::fixed, ios::floatfield );
			outfile.precision(1);
			outfile << confidence << " ";
			WritePose( outfile, constraint_mean );
			WriteCov(  outfile, constraint_covariance );
			outfile << endl;
		}

        CPosePDFGaussian GetPosePDFGaussian()
        {
			return CPosePDFGaussian(constraint_mean,constraint_covariance);
		}

        CPosePDFGaussian GetInvertedPosePDFGaussian()
        {
			CPosePDFGaussian inv;
			CPosePDFGaussian fwd(constraint_mean,constraint_covariance);
			fwd.inverse(inv);
			return inv;
		}

        inline const CPose2D& GetMean()
        {
			return constraint_mean;
		}

        inline const CMatrixDouble33 GetCovariance()
        {
			return constraint_covariance;
		}
	};

    class OccupancyTile
    {
    public:
		static const char WIDTH  = 32;
		static const char HEIGHT = 32;
        static const int  SIZE   = WIDTH * HEIGHT;
        static const char CELLS_PER_M = 20;
	private:
		const UUID submap_uuid;
		const CPoint2D top_left;

		IplImage* img;
		double ts;
	public:
		OccupancyTile( UUID _submap_uuid, CPoint2D _top_left, unsigned char *_occupancy_data,int data_len, int width,int height) :
		  submap_uuid(_submap_uuid), top_left(_top_left), ts(now())
		  {
			  //first expand bytes:
              unsigned char compressed[OccupancyTile::HEIGHT * OccupancyTile::WIDTH];
			  unsigned char *comp = &compressed[0];

              for (unsigned char* o = _occupancy_data; o <_occupancy_data + data_len; o++)
              {
				  comp[0] = (o[0]>>6)&3; //split bytes out
				  comp[1] = (o[0]>>4)&3; //split bytes out
				  comp[2] = (o[0]>>2)&3; //split bytes out
				  comp[3] = (o[0]>>0)&3; //split bytes out
				  comp+=4;
			  }
			  unsigned char *comp_end = comp;

			  //uncompress RLE data:
			  img = cvCreateImage(cvSize(width, height ),IPL_DEPTH_8U,1);  //should be byte aligned: no imageStep!

			  unsigned char *dst = (unsigned char*)&img->imageData[0];
			  unsigned char *dst_end = (unsigned char*)&img->imageData[width*height];

			  comp = &compressed[0];

              while ( (comp<comp_end) && (dst<dst_end) )
              {
                  if (comp[0]==3)
                  {
					  //this is a run
					  unsigned char val = comp[1] ? (comp[1]==1?127:255) : 0; //0=0, 1=127, 2=255
					  unsigned char count = (comp[2]<<6) + (comp[3]<<4) + (comp[4]<<2) + comp[5];

					  //unsigned char val = comp[1]<14 ? (comp[1]<<4) : 255; //round 14-15 back to 255
                      while ((count>0) && (dst<dst_end))
                      {
						  dst[0] = val; //set value
						  dst++;
						  count--;
					  }

					  comp+=6;
                  }
                  else
                  {
					  //normal
					  unsigned char val = comp[0] ? (comp[0]==1?127:255) : 0; //0=0, 1=127, 2=255
					  dst[0] = val;
					  dst++;
					  comp++;
				  }
			  }
		  }

          ~OccupancyTile(void)
          {
			  cvReleaseImage(&img);
		  }


          inline const UUID& GetSubmapUUID()
          {
			  return submap_uuid;
		  }

          inline const CPoint2D& GetTopLeft()
          {
			  return top_left;
		  }

          inline IplImage* GetData()
          {
			  return (IplImage*)img; //occupancy_data;
		  }

          inline const double& GetTimeStamp()
          {
			  return ts;
		  }
	};

    class Submap
    {
		UUID uuid;
		SubmapPose *pose;
		SubmapGroundTruth *ground_truth;
		std::list<SubmapConstraint*> base_constraints;  //this submap is the base of the constraint
		std::list<SubmapConstraint*> target_constraints;//this submap is the target of the constraint

		std::list<OccupancyTile*> new_occupancy_tiles;
		//std::map<int,OccupancyTile*> spatially_keyed_occupancy_tiles;
		//stdext::hash_map<UUID,HeightTile*> height_tiles;
		//CPosePDFGaussian robot_pose;

		CPoint2D extents_top_left;
		CPoint2D extents_bottom_right;

		int cell_count_occupied;
		int cell_count_free;
		int cell_count_unknown;

	public:

		int optimisation_id;
        int robot_id; //The numeric id of the robot which made this submap
		CPose2D *last_sent_pose;
		bool dont_match;

		//refactor these:
		double ts_occupancy_data; //number of 100-nanosecond intervals since January 1, 1601 (UTC).

		IplImage* occupancy_img; //image holding complete submap occupancy data
		mrpt::slam::COccupancyGridMap2D *occupancy_map_h;
		mrpt::slam::COccupancyGridMap2D *occupancy_map_v;
		bool occupancy_updated;

		Submap(void) :
		uuid(CreateUUID()),
			pose(NULL),
			ground_truth(NULL),
			extents_top_left(0,0),
			extents_bottom_right(0,0),
			last_sent_pose(NULL),
            dont_match(false),
            ts_occupancy_data(0),
            occupancy_img(NULL),
            occupancy_map_h(NULL),
            occupancy_map_v(NULL),
          occupancy_updated(true)
        {
		}

		Submap(const UUID& _uuid) :
			uuid(_uuid),
			pose(NULL),
			ground_truth(NULL),
			extents_top_left(0,0),
			extents_bottom_right(0,0),
			last_sent_pose(NULL),
            dont_match(false),
            ts_occupancy_data(0),
            occupancy_img(NULL),
            occupancy_map_h(NULL),
            occupancy_map_v(NULL),
            occupancy_updated(true)
        {
		}

		void AddUpennCost( CPosePDFGaussian &pose, double dt, std::vector<Vector3d> hlidar, std::vector<Vector3d> vlidar );
		int AddMichMapData( CPosePDFGaussian &pose, double dt,robot_map_data_t *map_data);
        void AddOccupancyGridData( double dt, const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid  );

        void WriteToAsciiStream( ofstream &outfile )
        {
			outfile.setf( ios::fixed, ios::floatfield );
			outfile.precision(2);
			outfile << "Submap " << uuid.c_str() << " ";
			outfile << cell_count_occupied << " " << cell_count_free << " " << cell_count_unknown << " ";
			outfile << extents_top_left.x() << " " << extents_top_left.y() << " ";
			outfile << extents_bottom_right.x() << " " << extents_bottom_right.y() << " ";
			outfile << ts_occupancy_data << endl;
		}

        inline const UUID& GetUUID()
        {
			return uuid;
		}

		void SetOccupancyTile( OccupancyTile* tile );
		void BuildSubmapData();

        inline IplImage* GetOccupancyData()
        {
			return occupancy_img;
		}

        inline bool GetOccupancyUpdatedFlag()
        {
			return occupancy_updated;
		}

        inline void ClearOccupancyUpdatedFlag()
        {
			occupancy_updated = false;
		}

        inline const CPoint2D& GetExtentsTopLeft()
        {
			return extents_top_left;
		}

        inline const CPoint2D& GetExtentsBottomRight()
        {
			return extents_bottom_right;
		}

        inline CPoint2D GetSize()
        {
			return CPoint2D( extents_bottom_right.x()-extents_top_left.x(),
				extents_top_left.y()-extents_bottom_right.y() );
		}

        inline int GetCellCountOccupied()
        {
			return cell_count_occupied;
		}

        inline int GetCellCountFree()
        {
			return cell_count_free;
		}

        inline int GetCellCountUnknown()
        {
			return cell_count_unknown;
		}

        inline bool IsConstrained()
        {
            //this submap has constraints applied to it's pose
			return base_constraints.size()>0 || target_constraints.size()>0 || ground_truth;
		}

        inline bool IsFree()
        {
			return !pose || (pose->fixed==false);
		}

        inline bool IsFixed()
        {
            //!= free
			return pose && pose->fixed;
		}

        ~Submap(void)
        {
			//delete uuid;
			//delete aggregated objects
			if (occupancy_img) cvReleaseImage(&occupancy_img);
			if (pose) delete pose;
			if (last_sent_pose) delete last_sent_pose;
			if (ground_truth) delete ground_truth;

			BOOST_FOREACH(mapping::SubmapConstraint* constraint, base_constraints)
                delete (SubmapConstraint*)constraint
                    ;
			BOOST_FOREACH(mapping::SubmapConstraint* constraint, target_constraints)
				delete (SubmapConstraint*)constraint;

			BOOST_FOREACH( OccupancyTile* tile, new_occupancy_tiles )
				delete tile;
		}

		//todo unhack back to ini
        #define SUBMAP_POSE_VERSION_MIN_LIN_DIST (0.1*10)
        #define SUBMAP_POSE_VERSION_MIN_ANG_DIST (0.29*10)

        bool SetPose(SubmapPose *_pose)
        {
            if ( uuid == _pose->GetUUID() )
            {
				//if the uuid's match, set this new pose
				bool changed = false;

                if (pose)
                {
                    //if an old pose existed, check if it's changed much:
					CPose2D delta = _pose->GetMean() - pose->GetMean();

					if (   (delta.norm()      < SUBMAP_POSE_VERSION_MIN_LIN_DIST)
						&& (fabs(delta.phi()) < SUBMAP_POSE_VERSION_MIN_ANG_DIST) )
						changed = false;
					else
						changed = true;
					delete pose;
				}
				pose = _pose;
				return changed;
			}
            else
            {
				return false;
				delete pose;
			}
		}

        inline SubmapPose* GetPose()
        {
			return pose;
		}

        inline const double& GetTimeStamp()
        {
			return ts_occupancy_data;
		}

        void SetGroundTruth(SubmapGroundTruth *gt)
        {
			if (ground_truth) delete ground_truth;
			ground_truth = gt;
		}

        inline SubmapGroundTruth* GetGroundTruth()
        {
			return ground_truth;
		}

        SubmapConstraint* GetSubmapConstraint( Submap* other_submap )
        {
			//return the constraint if it exists
			BOOST_FOREACH(SubmapConstraint* constraint, base_constraints)
				if (constraint->target_submap == other_submap)
					return constraint;

			BOOST_FOREACH(SubmapConstraint* constraint, target_constraints)
				if (constraint->base_submap == other_submap)
					return constraint;
			//otherwise, the constraint doesn't exist
			return 0;
		}

        std::list<Submap*> FindRecursive( std::list<Submap*> current_list, Submap* current_submap, Submap* target_submap, unsigned int max_depth)
        {
			current_list.push_back( current_submap );

			//check if search complete:
			if (current_submap == target_submap) //found target
				return current_list;

			std::list<Submap*> best_list;

			if (current_list.size()>=max_depth) //searched too far,
				return best_list; //return an empty list

			//recurse through its constraints:
            BOOST_FOREACH(SubmapConstraint* constraint, current_submap->base_constraints)
            {
				Submap* this_submap = constraint->target_submap;

				//skip submaps already in list
				list<Submap*>::iterator result = find(current_list.begin(), current_list.end(), this_submap);
				if (result != current_list.end()) continue;

				//recursive search:
				std::list<Submap*> this_best_list = FindRecursive( current_list, this_submap, target_submap, max_depth );
				//only complete or empty lists returned

				if (this_best_list.size())
					//a complete list was found to target
					if ((this_best_list.size()<best_list.size()) || (best_list.size()<1))
						best_list = this_best_list; //this list is better, set

				//return if no shorter path can be found:                
				if (best_list.size() && (best_list.size() <= (current_list.size()+1) )) return best_list;
			}

            BOOST_FOREACH(SubmapConstraint* constraint, current_submap->target_constraints)
            {
				Submap* this_submap = constraint->base_submap;
				//skip submaps already in list

				list<Submap*>::iterator result = find(current_list.begin(), current_list.end(), this_submap);
				if (result != current_list.end()) continue;

				//recursive search:
				std::list<Submap*> this_best_list = FindRecursive( current_list, this_submap, target_submap, max_depth );
				//only complete or empty lists returned

				if (this_best_list.size())
					//a complete list was found to target
					if ((this_best_list.size()<best_list.size()) || (best_list.size()<1))
						best_list = this_best_list; //this list is better, set

				//return if no shorter path can be found:
				if (best_list.size() && (best_list.size() <= (current_list.size()+1) )) return best_list;
			}
			return best_list;
		}

        SubmapConstraint* FindRecursiveSubmapConstraint( Submap* target_submap )
        {
			//recursive
			//todo search based on det(cov),
			//search for shortest list, yes this passes whole lists
			std::list<Submap*> submap_list;

            #define RECURSIVE_SEARCH_MAX_DEPTH 3
			submap_list = FindRecursive( submap_list, this, target_submap, RECURSIVE_SEARCH_MAX_DEPTH );

			if (submap_list.size()<1) //no path to target,
				return 0;

			SIMPLELOGGER_DEBUG( "Matching: FindRecursiveSubmapConstraint %d links.",(int)submap_list.size() );

			CPosePDFGaussian sum;
			Submap* last = this;
			submap_list.pop_front(); //skip first

            BOOST_FOREACH(Submap* submap, submap_list)
            {
				//shortest ordered list from this to target
				//compose PDF by summing constraints:
				SubmapConstraint* sc = last->GetSubmapConstraint(submap);
				ASSERT_(sc!=0);

                if ( last->GetUUID() == sc->base_submap_uuid )
                {
					sum += sc->GetPosePDFGaussian();
                }
                else
                {
					//this constraint needs inverting
					sum += sc->GetInvertedPosePDFGaussian();
				}

				last = submap;
			}

			//CPose2D mean = ; //target_submap->GetPose()->GetMean() - this->GetPose()->GetMean();
			//EEERRRGG todo check the mean is calculated from the current optimised layout, not the linked constraints:

			//return the result as a temporary submap constraint
			return new SubmapConstraint(string("temporary"),uuid,target_submap->GetUUID(),
					sum.mean,sum.cov,CONSTRAINT_TYPE_SUBMAP_MATCHER,0.5);
        }

        void RemoveSubmapConstraint( SubmapConstraint* constraint )
        {
			base_constraints.remove( constraint );
			target_constraints.remove( constraint );
		}

        void SetSubmapConstraint( SubmapConstraint* constraint )
        {
			if (constraint->base_submap == this)   base_constraints.push_back(   constraint );
			if (constraint->target_submap == this) target_constraints.push_back( constraint );
		}

        inline std::list<SubmapConstraint*>& GetBaseConstraints()
        {
			return base_constraints;  //this submap is the base of these constraints
		}

        inline std::list<SubmapConstraint*>& GetTargetConstraints()
        {
			return target_constraints;  //this submap is the target from these constraints
		}

        std::string toString()
        {
			std::ostringstream oss;
			if (pose)
				oss << "Submap: " << uuid.substr(0,6) << "_" << setiosflags(ios::fixed) << setprecision(1) << " x:" << pose->GetMean().x() << " y:" << pose->GetMean().y() << " phi:"<< R2D(pose->GetMean().phi()) <<" deg.";
			else
				oss << "Submap: " << uuid.substr(0,6) << "_";
			return oss.str();
		}

        friend std::ostream& operator<< (std::ostream& ostrm, Submap* submap)
        {
			//char uuid_str[50];
			//submap->GetUUID().GetAsString(uuid_str);
			if (submap->GetPose())
				ostrm << "Submap: " << submap->GetUUID().substr(0,6) << "_" << setiosflags(ios::fixed) << setprecision(1) << " x:" << submap->GetPose()->GetMean().x() << " y:" << submap->GetPose()->GetMean().y() << " phi:"<<submap->GetPose()->GetMean().phi();
			else
				ostrm << "Submap: " << submap->GetUUID().substr(0,6) << "_";
			return ostrm;
		}
	};

    class Map
    {
		//main interface to Map data structures
		//updates to map are queued and batch-updated in critical sections
	public:
		std::map<UUID,Submap*> submaps;
		std::list<Submap*> submaps_list;
		bool graph_changed;
                //calum QMutex lock_graph;
		Map() : graph_changed(true)
		{}

        inline std::list<Submap*>& GetSubmapList()
        {
			return submaps_list;
		}

		//set submap, without altering std::map<UUID,Submap*> map
        void SetSubmap( Submap* submap )
        {
			Submap *existing = GetSubmap( submap->GetUUID() );
			if (existing)
				submaps_list.remove(existing);
			submaps[submap->GetUUID()] = submap;
			submaps_list.push_back(submap);
			delete existing;
		}

		//get submap, without altering std::map<UUID,Submap*> map
        Submap* GetSubmap( const UUID& uuid )
        {
			map<UUID,Submap*>::iterator i = submaps.find(uuid);
            if (i == submaps.end())
            {
				return 0;
            }
            else
            {
				return i->second;
			}
		}

		//get or add submap
        Submap* GetOrCreateSubmap( const UUID& uuid )
        {
			Submap * submap;
			map<UUID,Submap*>::iterator i = submaps.find(uuid);
            if (i == submaps.end())
            {
				//not found, create
				submap = new Submap(uuid);
				submaps[uuid] = submap;
				submaps_list.push_back(submap);
            }
            else
            {
				//get existing submap
				submap = i->second;
			}
			return submap;
		}

		//add or replace submap pose
        void SetSubmapPose( mapping::SubmapPose * pose )
        {
			Submap *submap = GetOrCreateSubmap(pose->GetUUID());
            if ( pose->fixed || submap->IsFree() )
            {
                //this pose is from a GCS, or the current pose is not fixed
				graph_changed = submap->SetPose(pose) || graph_changed;
            }
            else
            {
				delete pose;
			}
		}

		//add or update submap occupancy tile
        void SetSubmapTile( mapping::OccupancyTile * tile )
        {
			Submap *submap = GetOrCreateSubmap(tile->GetSubmapUUID());
			submap->SetOccupancyTile( tile );
		}

		//add or update submap ground truth
        void SetSubmapGroundTruth( mapping::SubmapGroundTruth * ground_truth )
        {
			Submap *submap = GetOrCreateSubmap(ground_truth->GetSubmapUUID());
			submap->SetGroundTruth( ground_truth );
		}

        void DeleteSubmapGroundTruth( const UUID& uuid )
        {
			Submap *submap = GetOrCreateSubmap( uuid );
			submap->SetGroundTruth( 0 );
		}

		//add or replace submap constraint
        void SetSubmapConstraint( SubmapConstraint* new_constraint)
        {
			Submap *base_submap   = GetOrCreateSubmap( new_constraint->base_submap_uuid );
			Submap *target_submap = GetOrCreateSubmap( new_constraint->target_submap_uuid );

			//check if this base->target constraint already exists
			SubmapConstraint* existing_constraint = base_submap->GetSubmapConstraint( target_submap );

			//if the constraint doesnt exist, or if its uuid is alphabetically greater set it
			//---this enforces consistency if two robots create the same constraint at the same time.
			if ( (!existing_constraint)
				||(  existing_constraint->confidence<new_constraint->confidence )
				||(  (existing_constraint->confidence==new_constraint->confidence) &&(new_constraint->uuid > existing_constraint->uuid) ) )
			{
				//todo use setters
				new_constraint->base_submap = base_submap;
				new_constraint->target_submap = target_submap;

                if (existing_constraint)
                {
					base_submap->RemoveSubmapConstraint(existing_constraint);
					target_submap->RemoveSubmapConstraint(existing_constraint);
					delete existing_constraint;
				}

				base_submap->SetSubmapConstraint(new_constraint);
				target_submap->SetSubmapConstraint(new_constraint);
            }
            else
            {
				//better constraint already exists
				delete new_constraint;
			}
		}
	};
}
