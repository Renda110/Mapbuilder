#include "Mapping.h"
#include "MapBuilder.h"
#include <cv.h>
#include <highgui.h>
#include <uuid/uuid.h>
#include "zlib.h"
#include "opencv2/core/core.hpp"

using namespace cv;

#define MYWRITE_SUBMAPS (1);

namespace mapping
{
    string CreateUUID()
    {
        uuid_t myUUID;
        char uuid_string[37];
        uuid_generate(myUUID);
        uuid_unparse_upper(myUUID, uuid_string);

        return uuid_string;
    }

    //suitable for 100km mapping area:
    void Submap::SetOccupancyTile( OccupancyTile* tile )
    {
        //spatial lookup by top_left
        if (tile)
        {
			ts_occupancy_data = max( ts_occupancy_data, tile->GetTimeStamp() ); //remember most recent update
			new_occupancy_tiles.push_back(tile);
		}
	}

    #define round(x) ((x) >= 0 ? (int)((x) + 0.5) : (int)((x) - 0.5))

    void Submap::BuildSubmapData()
    {
		//only update submaps with new tiles
        if (new_occupancy_tiles.size() > 0)
        {
			//global extents of new tiles in cells
			//could do this in meters, however float-> int rounding issues = trouble
			int ntlx =  1000000000;  //new top left x coord
			int ntly = -1000000000;  //new top left y coord
			int nbrx = -1000000000;
			int nbry =  1000000000;

			//find extents of new tiles:
            BOOST_FOREACH( OccupancyTile* tile, new_occupancy_tiles )
            {
				int ttlx = round( tile->GetTopLeft().x()*OccupancyTile::CELLS_PER_M ); //round to nearest pixel
				int ttly = round( tile->GetTopLeft().y()*OccupancyTile::CELLS_PER_M );

				IplImage* tile_img = tile->GetData();

				ntlx = min( ttlx, ntlx );
				ntly = max( ttly, ntly );
				nbrx = max( ttlx + tile_img->width,  nbrx );
				nbry = min( ttly - tile_img->height, nbry );
			}

			ntlx -= 8; //give each submap some extra border
			ntly += 8; //to accommodate gaussian blur
			nbrx += 8; //must be multiple of 8: opencv image widthstep
			nbry -= 8;

			//find current extents:
			int ctlx = round( extents_top_left.x()*OccupancyTile::CELLS_PER_M ); //round to nearest pixel
			int ctly = round( extents_top_left.y()*OccupancyTile::CELLS_PER_M );
			int cbrx = ctlx;
			int cbry = ctly;

            if (occupancy_img)
            {
                //if data exists
				cbrx = ctlx + occupancy_img->width;
				cbry = ctly - occupancy_img->height;
			}

			//check if the current extents are too small
            if ((ntlx < ctlx) || (ntly > ctly) || (nbrx > cbrx) || (nbry < cbry))
            {
				//yes, new extents
				ntlx = min( ctlx, ntlx);
				ntly = max( ctly, ntly);
				nbrx = max( cbrx, nbrx);
				nbry = min( cbry, nbry);

				//create a new cv image
				IplImage* new_img = cvCreateImage(cvSize( nbrx-ntlx, ntly-nbry ),IPL_DEPTH_8U,3);
				memset( new_img->imageData, 127, new_img->width * new_img->height * 3);

				//if we've already got data, copy it over
                if (occupancy_img)
                {
					CvRect r = cvRect( ctlx-ntlx, ntly-ctly, occupancy_img->width, occupancy_img->height );
					cvSetImageROI(new_img,r);
					cvSetImageCOI(new_img,1); //copy blue channel only
					cvSetImageCOI(occupancy_img,1);
					cvCopy(occupancy_img,new_img); //copy data
					cvResetImageROI(new_img);
					cvSetImageCOI(new_img,0);
					cvReleaseImage(&occupancy_img);
				}

				occupancy_img = new_img;
				//set the new extents
				ctlx = ntlx;
				ctly = ntly;
				cbrx = nbrx;
				cbry = nbry;

				extents_top_left     = CPoint2D( (double)ctlx/OccupancyTile::CELLS_PER_M, (double)ctly/OccupancyTile::CELLS_PER_M );
				extents_bottom_right = CPoint2D( (double)cbrx/OccupancyTile::CELLS_PER_M, (double)cbry/OccupancyTile::CELLS_PER_M );
			}

			//we now have occupancy_img, correct size: blit new tiles into it
			//TODO verify the order this iterates
			cvSetImageCOI(occupancy_img,1); //blue channel only

            BOOST_FOREACH( OccupancyTile* tile, new_occupancy_tiles )
            {
				int ttlx = round( tile->GetTopLeft().x()*OccupancyTile::CELLS_PER_M ); //round to nearest pixel
				int ttly = round( tile->GetTopLeft().y()*OccupancyTile::CELLS_PER_M );

				IplImage* tile_img = tile->GetData();

//				int tx = ttlx - ntlx; //tile offset in pixels
//				int ty = ntly - ttly; //tile offset in pixels
//				for (int y=0; y<tile_img->height; y++) //copy tile data
//				memcpy( &occupancy_img->imageData[tx+(ty+y)*occupancy_img->widthStep ], &tile_img->imageData[y*OccupancyTile::WIDTH],OccupancyTile::WIDTH*sizeof(char));

                CvRect r = cvRect( ttlx-ctlx, ctly-ttly, tile_img->width, tile_img->height );

				cvSetImageROI(occupancy_img,r);
				cvCopy(tile_img,occupancy_img); //copy data

				delete tile; //we're done with the tile
			}

			cvSetImageCOI(occupancy_img,0);
			cvResetImageROI(occupancy_img);

			new_occupancy_tiles.clear(); //referenced tiles already deleted

			//premake Gaussian blurred version of image
			IplImage* temp_img        = cvCreateImage(cvSize( occupancy_img->width, occupancy_img->height), IPL_DEPTH_8U, 1);
			IplImage* thresholded_img = cvCreateImage(cvSize( occupancy_img->width, occupancy_img->height), IPL_DEPTH_8U, 1);
			IplImage* blurred_img     = cvCreateImage(cvSize( occupancy_img->width, occupancy_img->height), IPL_DEPTH_8U, 1);

			//copy occupancy grid
			cvSetImageCOI(occupancy_img,1 );
			cvCopy(occupancy_img, temp_img); //single channel needed for cvThreshold

			//binary threshold
			//also update cell counts
			cell_count_occupied = cell_count_free = cell_count_unknown = 0;

			//cvThreshold( temp_img, thresholded_img, 10, 255, CV_THRESH_BINARY_INV);
			unsigned char* occupancy = (unsigned char*) temp_img->imageData;
			unsigned char* threshold = (unsigned char*) thresholded_img->imageData;

            for (int i=0; i<temp_img->width*temp_img->height; i++)
            {
				//widthStep should be mult of 8 for opengl, ignore it
				assert(true); //check this

                if (occupancy[i]<30)
                {
					//occupied cell
					cell_count_occupied++;
					threshold[i] = 255;
                }
                else
                {
					//not occupied
                    if (occupancy[i]>225)
                    {
						//white freespace cell
						cell_count_free++;
						threshold[i] = 0;
                    }
                    else
                    {
						//grey
						cell_count_unknown++;
						threshold[i] = 0;
					}
				}
			}

			//gaussian blurred
			//todo this isn't used in matcher currently
			Mat temp = Mat(thresholded_img);
			Mat temp2 = Mat(blurred_img);
			GaussianBlur( temp, temp2, cvSize(9,9),0);

			cvSetImageCOI(occupancy_img,2 );//green channel
			cvCopy(blurred_img,occupancy_img);
			cvSetImageCOI(occupancy_img,3 );//red channel
			cvCopy(thresholded_img, occupancy_img);
			cvSetImageCOI(occupancy_img,0 );

			cvReleaseImage(&temp_img);
			cvReleaseImage(&thresholded_img);
			cvReleaseImage(&blurred_img);

			//dump submap tile to disk for persistence
            #ifdef MYWRITE_SUBMAPS
                char submap_filename[255];
                sprintf(submap_filename, "%s/submaps/%s.png", DATA_PATH, uuid.c_str() );

                cvSaveImage(submap_filename, occupancy_img);

                SIMPLELOGGER_ERROR("Submap printed");
            #endif

			//set flag
			occupancy_updated = true;
		}
	}

	void Submap::AddUpennCost( CPosePDFGaussian &pose, double dt, std::vector<Vector3d> hlidar, std::vector<Vector3d> vlidar )
	{
		int NH = (int)hlidar.size();
		int NV = (int)vlidar.size();
		double min_x=999999,max_x=-999999,min_y=9999999,max_y=-99999;

        for (int i=0; i<NH; i++)
        {
			CPoint2D p =  pose.mean + CPoint2D(hlidar.at(i)[0], hlidar.at(i)[1]); //x,y are in robocentric coords
			hlidar.at(i)[0] = p.x();  //convert to submap coords and save
			hlidar.at(i)[1] = p.y();
			min_x = MIN(p.x()-0.2, min_x);
			max_x = MAX(p.x()+0.2, max_x);
			min_y = MIN(p.y()-0.2, min_y);
			max_y = MAX(p.y()+0.2, max_y);
		}

        for (int i=0; i<NV; i++)
        {
			CPoint2D p =  pose.mean + CPoint2D(vlidar.at(i)[0], vlidar.at(i)[1]); //x,y are in robocentric coords
			vlidar.at(i)[0] = p.x();  //convert to submap coords and save
			vlidar.at(i)[1] = p.y();
			min_x = MIN(p.x()-0.2, min_x);
			max_x = MAX(p.x()+0.2, max_x);
			min_y = MIN(p.y()-0.2, min_y);
			max_y = MAX(p.y()+0.2, max_y);
		}

		min_x = (double)(((int)((min_x+0.0001)*OccupancyTile::CELLS_PER_M)/8)*8-8)/OccupancyTile::CELLS_PER_M;  //round to 8 pixel boundaries
		max_x = (double)(((int)((max_x+0.0001)*OccupancyTile::CELLS_PER_M)/8)*8+8)/OccupancyTile::CELLS_PER_M;
		min_y = (double)(((int)((min_y+0.0001)*OccupancyTile::CELLS_PER_M)/8)*8-8)/OccupancyTile::CELLS_PER_M;
		max_y = (double)(((int)((max_y+0.0001)*OccupancyTile::CELLS_PER_M)/8)*8+8)/OccupancyTile::CELLS_PER_M;

        if (occupancy_map_h)
        {
			//resize existing
			min_x = MIN( occupancy_map_h->getXMin(), min_x);
			max_x = MAX( occupancy_map_h->getXMax(), max_x);
			min_y = MIN( occupancy_map_h->getYMin(), min_y);
			max_y = MAX( occupancy_map_h->getYMax(), max_y);;
			occupancy_map_h->resizeGrid(min_x,max_x,min_y,max_y, 0.5f, false);
			occupancy_map_v->resizeGrid(min_x,max_x,min_y,max_y, 0.5f, false);
        }
        else
        {
			occupancy_map_h = new COccupancyGridMap2D(min_x,max_x,min_y,max_y, 0.1f);
			occupancy_map_v = new COccupancyGridMap2D(min_x,max_x,min_y,max_y, 0.1f);
		}

 		extents_top_left     = CPoint2D( min_x,max_y );
		extents_bottom_right = CPoint2D( max_x,min_y );
		ts_occupancy_data = dt;//*TS_SEC;// now();

		// For updateCell_fast methods:
		COccupancyGridMap2D::cellType  *theMapArray_h = occupancy_map_h->getRow(0);
		COccupancyGridMap2D::cellType  *theMapArray_v = occupancy_map_v->getRow(0);
		unsigned  theMapSize_x = occupancy_map_h->getSizeX();

		float maxCertainty = 0.4;
		COccupancyGridMap2D::cellType    logodd_observation  = 20;//occupancy_map_h->p2l(maxCertainty);
		COccupancyGridMap2D::cellType    logodd_observation_occupied = 50;//3*logodd_observation;

        if (logodd_observation<=0)
        {
            logodd_observation=1;
        }

        COccupancyGridMap2D::cellType    logodd_thres_occupied = OCCGRID_CELLTYPE_MIN+logodd_observation_occupied;
		COccupancyGridMap2D::cellType    logodd_thres_free     = OCCGRID_CELLTYPE_MAX-logodd_observation;

		int cx, cy;
		int cx0 = occupancy_map_h->x2idx( pose.mean.x() ); //laser pose
		int cy0 = occupancy_map_h->y2idx( pose.mean.y() );

		// Insert hlidar free cells
        for (int i=0;i<NH;i++)
        {
            //NH
            if ( hlidar.at(i)[2] >-50)
            {
                // Starting position: Laser position
                cx = cx0;
                cy = cy0;

                // Target, in cell indexes:
                int trg_cx = occupancy_map_h->x2idx( hlidar.at(i)[0] );
                int trg_cy = occupancy_map_h->y2idx( hlidar.at(i)[1] );

                // Use "fractional integers" to approximate float operations
                //  during the ray tracing:
                int Acx  = trg_cx - cx;
                int Acy  = trg_cy - cy;

                int Acx_ = abs(Acx);
                int Acy_ = abs(Acy);

                int nStepsRay = max( Acx_, Acy_ );
                if (!nStepsRay) continue; // May be...

                // Integers store "float values * 128"
                float  N_1 = 1.0f / nStepsRay;   // Avoid division twice.

                // Increments at each raytracing step:
                #define FRBITS	9
                int  frAcx = round( (Acx<< FRBITS) * N_1 );  //  Acx*128 / N
                int  frAcy = round( (Acy<< FRBITS) * N_1 );  //  Acy*128 / N

                int frCX = cx << FRBITS;
                int frCY = cy << FRBITS;

                for (int nStep = 0;nStep<nStepsRay;nStep++)
                {
                    occupancy_map_h->updateCell_fast_free(cx,cy, logodd_observation, logodd_thres_free, theMapArray_h, theMapSize_x );
                    frCX += frAcx;
                    frCY += frAcy;
                    cx = frCX >> FRBITS;
                    cy = frCY >> FRBITS;
                }
            }
		}  // End of each point

                //hlidar obstacles
        for (int i=0;i<NH;i++)
        {
            //NH
			int trg_cx = occupancy_map_h->x2idx( hlidar.at(i)[0] );
			int trg_cy = occupancy_map_h->y2idx( hlidar.at(i)[1] );
                        // And finally, the occupied cell at the end:
			if (abs(hlidar.at(i)[2])>-50)
            {
				occupancy_map_h->updateCell_fast_occupied(trg_cx,trg_cy, logodd_observation_occupied, logodd_thres_occupied, theMapArray_h, theMapSize_x );
             }
        }

		// Insert vlidar without raytracing:
        for (int i=0;i<NV;i++)
        {
			int trg_cx = occupancy_map_v->x2idx( vlidar.at(i)[0] );
			int trg_cy = occupancy_map_v->y2idx( vlidar.at(i)[1] );
			if (vlidar.at(i)[2]>50)
            {
				occupancy_map_v->updateCell_fast_occupied(trg_cx,trg_cy, logodd_observation_occupied, logodd_thres_occupied, theMapArray_v, theMapSize_x );
            }
            else if (vlidar.at(i)[2]<-10)
            {
				occupancy_map_v->updateCell_fast_free(trg_cx,trg_cy, logodd_observation, logodd_thres_free, theMapArray_v, theMapSize_x );
            }
        }

		mrpt::utils::CImage cimg_h;
		occupancy_map_h->getAsImage(cimg_h);
		IplImage *occupancy_data_h = (IplImage *)cimg_h.getAsIplImage();

		mrpt::utils::CImage cimg_v;
		occupancy_map_v->getAsImage(cimg_v);
		IplImage *occupancy_data_v = (IplImage *)cimg_v.getAsIplImage();

        if (occupancy_img)
        {
			cvReleaseImage(&occupancy_img);
		}

		occupancy_img             = cvCreateImage(cvSize( occupancy_data_h->width, occupancy_data_h->height), IPL_DEPTH_8U, 3);
		IplImage* blurred_img     = cvCreateImage(cvSize( occupancy_data_h->width, occupancy_data_h->height), IPL_DEPTH_8U, 1);
		IplImage* thresholded_img = cvCreateImage(cvSize( occupancy_data_h->width, occupancy_data_h->height), IPL_DEPTH_8U, 1);

		//binary threshold
		//also update cell counts
		cell_count_occupied = cell_count_free = cell_count_unknown = 0;
		//cvThreshold( temp_img, thresholded_img, 10, 255, CV_THRESH_BINARY_INV);
		unsigned char* occupancy = (unsigned char*) occupancy_data_h->imageData;
		unsigned char* threshold = (unsigned char*) thresholded_img->imageData;

        for (int i=0; i<occupancy_data_h->width*occupancy_data_h->height; i++)
        {
			//widthStep should be mult of 8 for opengl, ignore it
            if (occupancy[i]<30)
            {
				//occupied cell
				cell_count_occupied++;
				threshold[i] = 255;
            }
            else
            {
				//not occupied
                if (occupancy[i]>225)
                {
					//white freespace cell
					cell_count_free++;
					threshold[i] = 0;
                }
                else
                {
					//grey
					cell_count_unknown++;
					threshold[i] = 0;
				}
			}
		}

		//gaussian blurred
		//todo this isn't used in matcher currently
		Mat temp = Mat(thresholded_img);
		Mat temp2 = Mat(blurred_img);
		GaussianBlur( temp, temp2, cvSize(9,9),0);

		cvSetImageCOI(occupancy_img,1 );//blue channel
		cvCopy(occupancy_data_h,occupancy_img);
		cvSetImageCOI(occupancy_img,2 );//green channel
		cvCopy(blurred_img,occupancy_img);
		cvSetImageCOI(occupancy_img,3 );//red channel
		cvCopy(occupancy_data_v, occupancy_img);
		cvSetImageCOI(occupancy_img,0 );

		cvReleaseImage(&blurred_img);
		cvReleaseImage(&thresholded_img);

        #ifdef WRITE_SUBMAPS
            //dump submap tile to disk for persistence
            char submap_filename[255];
            sprintf(submap_filename, "%s/submaps/%s.png", DATA_PATH, uuid.c_str() );
            //sprintf(submap_filename, "%s/submaps/%s-%d.png", Configuration::getInstance()->getDataPath(), uuid.c_str(), submapver++ );
            cvSaveImage(submap_filename,occupancy_img);
        #endif

                //set flag
		occupancy_updated = true;
	}

    int Submap::AddMichMapData( CPosePDFGaussian &pose, double dt,robot_map_data_t *map_data)
    {
        //decode map data
        grid_map_t *gm = &map_data->gridmap; // Local map data near this pose in LOCAL frame
        CPose2D local_pose(map_data->xyt_local[0],map_data->xyt_local[1],map_data->xyt_local[2]);

        if (gm->encoding != grid_map_t::ENCODING_GZIP)
        {
            SIMPLELOGGER_ERROR("gm->encoding != grid_map_t::ENCODING_GZIP)");
            return(-1); //skip if encoded incorrectly
        }
        //assume gzip encoded map, decode:
        //in java: gm = MagicUtil.decodeGZIP(gm);

        //buffer for ungzipped data:
        size_t gm_decompressed_len = gm->width*gm->height;
        vector<uint8_t> gm_decompressed(gm_decompressed_len);

        z_stream strm;
        strm.zalloc = 0;
        strm.zfree = 0;
        strm.next_in =  (uchar*)&gm->data[0];
        strm.avail_in = gm->datalen;
        strm.next_out = (uchar*)&gm_decompressed[0];
        strm.avail_out = gm_decompressed_len;

        // Add 32 to windowBits to enable zlib and
        //gzip decoding with automatic header detection
        if (inflateInit2(&strm,15+32) != Z_OK)
        {
            SIMPLELOGGER_ERROR("inflateInit2(&strm,15+32) != Z_OK");
            inflateEnd(&strm);
            return(-1);
        }

        if (inflate(&strm,Z_FINISH)!= Z_STREAM_END)
        {
            SIMPLELOGGER_ERROR("inflate(&strm,Z_FINISH)!= Z_STREAM_END");
            inflateEnd(&strm);
            return(-1);
        }

        inflateEnd(&strm);

        SIMPLELOGGER_DEBUG_MORE("map_data %dx%d %d==%d origin=%1.2f,%1.2f",  gm->width,gm->height,(int)gm_decompressed_len, (int)gm_decompressed_len - strm.avail_out, gm->x0,gm->y0  );

        //extract list of obstacle coordinates in submap frame:
        std::vector<CPoint2D> hlidar;

        double min_x=999999,max_x=-999999,min_y=9999999,max_y=-99999;

        for (int ly = 0; ly < gm->height; ly++)
        {
            for(int lx = 0; lx <gm->width; lx++)
            {
                uchar lv = gm_decompressed[ly*gm->width + lx];

                if (lv == 255)
                {
                    //is obstacle
                    //mich local coordinates of pixel:
                    double px = gm->x0 + (lx + 0.5)*gm->meters_per_pixel;
                    double py = gm->y0 + (ly + 0.5)*gm->meters_per_pixel;
                    CPoint2D p(px,py);
                    //robocentric coordinates:
                    CPoint2D p_r = p-local_pose;
                    //coordinate in current submap:
                    CPoint2D p_s = pose.mean + p_r;
                    hlidar.push_back( p_s );

                    min_x = MIN(p_s.x(), min_x);
                    max_x = MAX(p_s.x(), max_x);
                    min_y = MIN(p_s.y(), min_y);
                    max_y = MAX(p_s.y(), max_y);
                }
            }
        }

        int NH = hlidar.size();

        min_x = (double)(((int)((min_x-0.2+0.0001)*OccupancyTile::CELLS_PER_M)/8)*8-8)/OccupancyTile::CELLS_PER_M;  //round to 8 pixel boundaries
        max_x = (double)(((int)((max_x+0.2+0.0001)*OccupancyTile::CELLS_PER_M)/8)*8+8)/OccupancyTile::CELLS_PER_M;
        min_y = (double)(((int)((min_y-0.2+0.0001)*OccupancyTile::CELLS_PER_M)/8)*8-8)/OccupancyTile::CELLS_PER_M;
        max_y = (double)(((int)((max_y+0.2+0.0001)*OccupancyTile::CELLS_PER_M)/8)*8+8)/OccupancyTile::CELLS_PER_M;

        if (occupancy_map_h)
        {
            //resize existing
            min_x = MIN( occupancy_map_h->getXMin(), min_x);
            max_x = MAX( occupancy_map_h->getXMax(), max_x);
            min_y = MIN( occupancy_map_h->getYMin(), min_y);
            max_y = MAX( occupancy_map_h->getYMax(), max_y);
            //printf("%f %f %f %f\n",min_x,max_x,min_y,max_y );
            occupancy_map_h->resizeGrid(min_x,max_x,min_y,max_y, 0.5f, false);
        }
        else
        {
            occupancy_map_h = new COccupancyGridMap2D(min_x,max_x,min_y,max_y, 0.1f);
        }

        extents_top_left     = CPoint2D( min_x,max_y );
        extents_bottom_right = CPoint2D( max_x,min_y );
        ts_occupancy_data = dt;//*TS_SEC;// now();

        // For updateCell_fast methods:
        COccupancyGridMap2D::cellType  *theMapArray_h = occupancy_map_h->getRow(0);
        unsigned  theMapSize_x = occupancy_map_h->getSizeX();

        float maxCertainty = 0.4;
        COccupancyGridMap2D::cellType    logodd_observation  = occupancy_map_h->p2l(maxCertainty);
        COccupancyGridMap2D::cellType    logodd_observation_occupied = 3*logodd_observation;
        if (logodd_observation<=0) logodd_observation=1;
        COccupancyGridMap2D::cellType    logodd_thres_occupied = OCCGRID_CELLTYPE_MIN+logodd_observation_occupied;
        COccupancyGridMap2D::cellType    logodd_thres_free     = OCCGRID_CELLTYPE_MAX-logodd_observation;

        int cx, cy;
        int cx0 = occupancy_map_h->x2idx( pose.mean.x() ); //laser pose
        int cy0 = occupancy_map_h->y2idx( pose.mean.y() );

        // Insert hlidar free cells
        for (int i=0;i<NH;i++)
        {
            // Starting position: Laser position
            cx = cx0;
            cy = cy0;

            // Target, in cell indexes:
            int trg_cx = occupancy_map_h->x2idx( hlidar.at(i).x() );
            int trg_cy = occupancy_map_h->y2idx( hlidar.at(i).y() );

            // Use "fractional integers" to approximate float operations
            //  during the ray tracing:
            int Acx  = trg_cx - cx;
            int Acy  = trg_cy - cy;

            int Acx_ = abs(Acx);
            int Acy_ = abs(Acy);

            int nStepsRay = max( Acx_, Acy_ );
            if (!nStepsRay) continue; // May be...

            // Integers store "float values * 128"
            float  N_1 = 1.0f / nStepsRay;   // Avoid division twice.

            // Increments at each raytracing step:
            #define FRBITS	9
            int  frAcx = round( (Acx<< FRBITS) * N_1 );  //  Acx*128 / N
            int  frAcy = round( (Acy<< FRBITS) * N_1 );  //  Acy*128 / N

            int frCX = cx << FRBITS;
            int frCY = cy << FRBITS;

            for (int nStep = 0;nStep<nStepsRay;nStep++)
            {
                    occupancy_map_h->updateCell_fast_free(cx,cy, logodd_observation, logodd_thres_free, theMapArray_h, theMapSize_x );
                    frCX += frAcx;
                    frCY += frAcy;
                    cx = frCX >> FRBITS;
                    cy = frCY >> FRBITS;
            }
        }  // End of each point

        //hlidar obstacles
        for (int i=0;i<NH;i++)
        {
            //NH
            int trg_cx = occupancy_map_h->x2idx( hlidar.at(i).x() );
            int trg_cy = occupancy_map_h->y2idx( hlidar.at(i).y() );
            occupancy_map_h->updateCell_fast_occupied(trg_cx,trg_cy, logodd_observation_occupied, logodd_thres_occupied, theMapArray_h, theMapSize_x );
        }

        mrpt::utils::CImage cimg_h;
        occupancy_map_h->getAsImage(cimg_h);
        IplImage *occupancy_data_h = (IplImage *)cimg_h.getAsIplImage();

        if (occupancy_img)
        {
                cvReleaseImage(&occupancy_img);
        }

        occupancy_img             = cvCreateImage(cvSize( occupancy_data_h->width, occupancy_data_h->height), IPL_DEPTH_8U, 3);
        IplImage* blurred_img     = cvCreateImage(cvSize( occupancy_data_h->width, occupancy_data_h->height), IPL_DEPTH_8U, 1);
        IplImage* thresholded_img = cvCreateImage(cvSize( occupancy_data_h->width, occupancy_data_h->height), IPL_DEPTH_8U, 1);
        //printf("occupancy_data %d %d \n",occupancy_data->width, occupancy_data->height );

        //binary threshold
        //also update cell counts
        cell_count_occupied = cell_count_free = cell_count_unknown = 0;

        //cvThreshold( temp_img, thresholded_img, 10, 255, CV_THRESH_BINARY_INV);
        unsigned char* occupancy = (unsigned char*) occupancy_data_h->imageData;
        unsigned char* threshold = (unsigned char*) thresholded_img->imageData;

        for (int i=0; i<occupancy_data_h->width*occupancy_data_h->height; i++)
        {
            //widthStep should be mult of 8 for opengl, ignore it
            if (occupancy[i]<30)
            {
                //occupied cell
                cell_count_occupied++;
                threshold[i] = 255;
            }
            else
            {
                //not occupied
                if (occupancy[i]>225)
                {
                        //white freespace cell
                        cell_count_free++;
                        threshold[i] = 0;
                }
                else
                {
                        //grey
                        cell_count_unknown++;
                        threshold[i] = 0;
                }
            }
        }

        //gaussian blurred
        //todo this isn't used in matcher currently
        Mat temp = Mat(thresholded_img);
        Mat temp2 = Mat(blurred_img);
        GaussianBlur( temp, temp2, cvSize(9,9),0);

        cvSetImageCOI(occupancy_img,1 );//blue channel
        cvCopy(occupancy_data_h,occupancy_img);
        cvSetImageCOI(occupancy_img,2 );//green channel
        cvCopy(blurred_img,occupancy_img);
        cvSetImageCOI(occupancy_img,3 );//red channel
        cvCopy(occupancy_data_h, occupancy_img);
        cvSetImageCOI(occupancy_img,0 );

        cvReleaseImage(&blurred_img);
        cvReleaseImage(&thresholded_img);

        #ifdef WRITE_SUBMAPS
            //dump submap tile to disk for persistence
            char submap_filename[255];
            sprintf(submap_filename, "%s/submaps/%s.png", DATA_PATH, uuid.c_str() );
            //sprintf(submap_filename, "%s/submaps/%s-%d.png", Configuration::getInstance()->getDataPath(), uuid.c_str(), submapver++ );
            cvSaveImage(submap_filename, occupancy_img);
        #endif

        //set flag
        occupancy_updated = true;
	}

    void Submap::AddOccupancyGridData( double dt, const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid  )
    {
        int w = grid->info.width;
        int h = grid->info.height;

        ROS_INFO("OccupancyGrid %dx%d origin=%1.2f,%1.2f",  w,h,
                  grid->info.origin.position.x, grid->info.origin.position.y);

        ROS_ASSERT( fabs( grid->info.resolution - 1.0/OccupancyTile::CELLS_PER_M ) < 1e-3 );
        ROS_ASSERT( w > 0 );
        ROS_ASSERT( h > 0 );
        ROS_ASSERT( (w/8)*8  == w);
        ROS_ASSERT( (h/8)*8 == h);

        extents_top_left     = CPoint2D( grid->info.origin.position.x,
                                         grid->info.origin.position.y+(double)h/OccupancyTile::CELLS_PER_M );
        extents_bottom_right = CPoint2D( grid->info.origin.position.x+(double)w/OccupancyTile::CELLS_PER_M,
                                         grid->info.origin.position.y );

        ts_occupancy_data = dt;

        if (occupancy_img)
        {
                cvReleaseImage(&occupancy_img);
        }

        occupancy_img             = cvCreateImage(cvSize( w, h), IPL_DEPTH_8U, 3);
        IplImage* occupancy_data_h= cvCreateImage(cvSize( w, h), IPL_DEPTH_8U, 1);
        IplImage* blurred_img     = cvCreateImage(cvSize( w, h), IPL_DEPTH_8U, 1);
        IplImage* thresholded_img = cvCreateImage(cvSize( w, h), IPL_DEPTH_8U, 1);

        //copy OccupancyGrid data to occupancy_img
        //binary threshold
        //also update cell counts
        cell_count_occupied = cell_count_free = cell_count_unknown = 0;

        for (int y=0; y<h; y++)
        {
            unsigned char* occupancy = (unsigned char*) occupancy_data_h->imageData + thresholded_img->widthStep*y;
            unsigned char* threshold = (unsigned char*) thresholded_img->imageData + thresholded_img->widthStep*y;

            for (int x=0; x<w; x++)
            {
                int8_t occ = grid->data[ (h-1-y)*w + x ];

                //widthStep should be mult of 8 for opengl, ignore it
                if (occ>50)
                {
                    //occupied cell
                    cell_count_occupied++;
                    threshold[x] = 255; //255
                    occupancy[x] = 0;
                }
                else
                {
                    //not occupied
                    if (occ==0)
                    {
                        //white freespace cell
                        cell_count_free++;
                        threshold[x] = 0;
                        occupancy[x] = 255;

                        /*switch (robot_id)
                        {
                        case 1:
                            occupancy[x] = 50;
                            break;
                        case 2:
                            occupancy[x] = 100;
                            break;
                        case 3:
                            occupancy[x] = 150;
                            break;
                        case 4:
                            occupancy[x] = 200;
                            break;
                        case 5:
                            occupancy[x] = 255;
                            break;
                        }*/

                    }
                    else
                    {
                        //grey unknown
                        cell_count_unknown++;
                        threshold[x] = 0;
                        occupancy[x] = 127; //127
                    }
                }
            }
        }

        //gaussian blurred
        //todo this isn't used in matcher currently
        Mat temp = Mat(thresholded_img);
        Mat temp2 = Mat(blurred_img);
        GaussianBlur( temp, temp2, cvSize(9,9),0);

        cvSetImageCOI(occupancy_img,1 );//blue channel
        cvCopy(occupancy_data_h,occupancy_img);
        cvSetImageCOI(occupancy_img,2 );//green channel
        cvCopy(blurred_img,occupancy_img);
        cvSetImageCOI(occupancy_img,3 );//red channel
        cvCopy(occupancy_data_h, occupancy_img);
        cvSetImageCOI(occupancy_img,0 );

        cvReleaseImage(&occupancy_data_h);
        cvReleaseImage(&blurred_img);
        cvReleaseImage(&thresholded_img);

        #ifdef WRITE_SUBMAPS
            //dump submap tile to disk for persistence
            char submap_filename[255];
            sprintf(submap_filename, "%s/submaps/%s.png", DATA_PATH, uuid.c_str() );
            //sprintf(submap_filename, "%s/submaps/%s-%d.png", Configuration::getInstance()->getDataPath(), uuid.c_str(), submapver++ );
            cvSaveImage(submap_filename, occupancy_img);
        #endif

        //set flag
        occupancy_updated = true;
    }
}
