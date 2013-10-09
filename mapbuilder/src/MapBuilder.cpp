#include "MapBuilder.h"

#include <cv.h>
#include <highgui.h>
#include <uuid/uuid.h>
#include <unistd.h>
#include <stdio.h>
#include <mrpt/base/include/mrpt/utils/CTicTac.h>
#include <list>
#include <mrpt/base/include/mrpt/poses/CPoseOrPoint.h>
#include <mrpt/base/include/mrpt/poses/CPose2D.h>

#include <fstream>
#include <iostream>

//#define TS_SEC (10000000)

using namespace mrpt;
using namespace mrpt::poses;
using namespace mrpt::system;
using namespace mrpt::random;
using namespace std;
using namespace Eigen;
using namespace sba;

extern bool is_GCS;
extern bool is_WAMBOT;
extern bool is_GCS_OPTIMISER;
extern bool is_GCS_MATCHER;

namespace mapping
{
    bool MapBuilder::CheckOpenGlError(char *extra)
    {
        GLenum errCode;
        const GLubyte *errString;

        if ((errCode = glGetError()) != GL_NO_ERROR)
        {
            errString = gluErrorString(errCode);
            SIMPLELOGGER_ERROR("OpenGL %s Error: %s", extra, errString);
            return false;
        }
        return true;
    }

    void MapBuilder::Initialize(char *config_filename)
    {
        //config file:
        sprintf(config_filepath, "%s/%s", CONFIG_PATH, config_filename);

        SIMPLELOGGER_INFO("Initialising... %s", config_filepath);

        //maps are written to home/output/maps
        sprintf(submaps_path, "%s/submaps", DATA_PATH);
        sprintf(maps_path, "%s/maps", DATA_PATH);
        sprintf(map_state_path, "%s/map_state", DATA_PATH);
        sprintf(matches_path, "%s/submaps", DATA_PATH);

        createDirectory(DATA_PATH);
        createDirectory(submaps_path);
        createDirectory(maps_path);
        createDirectory(map_state_path);

        //remember if we're the GCS
        is_GCS = true;

        if (!fileExists(config_filepath))
        {
            SIMPLELOGGER_ERROR("Configuration file %s not found.", config_filepath);
            //exit(-1);
        }

        CConfigFile iniFile(config_filepath);

        //config options:
        MRPT_LOAD_CONFIG_VAR(GENERATED_MAP_FORMAT, string, iniFile, "MappingApplication")
        MRPT_LOAD_CONFIG_VAR(generatedMapSizeMultiple, int, iniFile, "MappingApplication")
        MRPT_LOAD_CONFIG_VAR(submapMatchingMinimumOccupiedCellCount, int, iniFile, "MappingApplication")
        MRPT_LOAD_CONFIG_VAR(submapMatchingScoreThreshold, double, iniFile, "MappingApplication")
        MRPT_LOAD_CONFIG_VAR(submapMatchingScoreThresholdWithPrior, double, iniFile, "MappingApplication")
        MRPT_LOAD_CONFIG_VAR_DEGREES(REFERENCE_POSITION_LATITUDE, iniFile, "MappingApplication")
        MRPT_LOAD_CONFIG_VAR_DEGREES(REFERENCE_POSITION_LONGITUDE, iniFile, "MappingApplication")

        MRPT_LOAD_CONFIG_VAR(SUBMAP_POSE_CORRECTION_MIN_LIN_DIST, double, iniFile, "MappingApplication")
        MRPT_LOAD_CONFIG_VAR_DEGREES(SUBMAP_POSE_CORRECTION_MIN_ANG_DIST, iniFile, "MappingApplication")

        MRPT_LOAD_CONFIG_VAR(HMI_GROUND_TRUTH_CONSTRAINT_ERROR_LIN_STDEV, double, iniFile, "MappingApplication")
        MRPT_LOAD_CONFIG_VAR_DEGREES(HMI_GROUND_TRUTH_CONSTRAINT_ERROR_ANG_STDEV, iniFile, "MappingApplication")

        double northing, easting; //set utm zone
        LatLonToUtmWGS84(REFERENCE_POSITION_LATITUDE, REFERENCE_POSITION_LONGITUDE, northing, easting, utm_zone);
        SIMPLELOGGER_INFO("Set UTM grid zone to %s", utm_zone);
        reference_position = CPose2D(easting, northing, 0);

        //setup opengl
        SIMPLELOGGER_INFO("Initialising OpenGL...");

        GLenum err = glewInit();

        if (GLEW_OK != err)
        {
            SIMPLELOGGER_ERROR("Can't init GLEW.");
            exit(-1);
        }

        //different buffer size for GCS/ Wambots
        if (is_GCS)
        {
            map_pixel_width = 5120;
            map_pixel_height = 5120;
        }
        else
        {
            map_pixel_width = 2048;
            map_pixel_height = 2048;
        }

        matching_pixel_width = 150;
        matching_pixel_height = 150;

        //////////Set up an FBO with one renderbuffer attachment
        map_framebuffer = 0;
        map_renderbuffers[0] = 0;
        map_renderbuffers[1] = 0;

        glGenFramebuffersEXT(1, &map_framebuffer); //Generate a new framebuffer id
        CheckOpenGlError("glGenFramebuffersEXT");

        if (!map_framebuffer)
        {
            exit(-1);
        }

        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, map_framebuffer); //Bind to it
        CheckOpenGlError("glBindFramebufferEXT map_framebuffer");

        //Generate 2 new renderbuffer ids one for the color data, and one for a depth buffer
        glGenRenderbuffersEXT(2, map_renderbuffers);

        if (!CheckOpenGlError("glGenRenderbuffersEXT") || !map_renderbuffers[0] || !map_renderbuffers[1])
        {
            SIMPLELOGGER_ERROR("Can't get render buffer ID's");
            exit(-1);
        }

        //bind a depth buffer to the FBO
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, map_renderbuffers[1]);
        CheckOpenGlError("glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, map_renderbuffers[1])");

        glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, map_pixel_width, map_pixel_height);
        CheckOpenGlError("glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, save_map_pixel_width, save_height)");

        glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, map_renderbuffers[1]);
        CheckOpenGlError("glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, renderbuffers[1])");

        GLuint map_texture;
        glGenTextures(1, &map_texture);
        glBindTexture(GL_TEXTURE_2D, map_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, map_pixel_width, map_pixel_height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, map_texture, 0);
        CheckOpenGlError("glFramebufferTexture2DEXT");

        GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
        //make sure the framebuffer is complete

        if (status != GL_FRAMEBUFFER_COMPLETE_EXT)
        {
            SIMPLELOGGER_ERROR("Can't initialise OpenGL map render buffer (%dx%d pixels).", map_pixel_width, map_pixel_height);
            exit(-1);
        }
        SIMPLELOGGER_INFO("Initialised map render buffer (%dx%d pixels).", map_pixel_width, map_pixel_height);

        //////////Set up another FBO for matching results
        matching_framebuffer = 0;
        matching_renderbuffer = 0;

        glGenFramebuffersEXT(1, &matching_framebuffer); //Generate a new framebuffer id
        CheckOpenGlError("glGenFramebuffersEXT");

        if (!matching_framebuffer)
        {
            exit(-1);
        }

        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, matching_framebuffer); //Bind to it
        CheckOpenGlError("glBindFramebufferEXT matching_framebuffer");

        glGenTextures(1, &matching_renderbuffer);
        glBindTexture(GL_TEXTURE_2D, matching_renderbuffer);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, matching_pixel_width, matching_pixel_height, 0, GL_RGBA, GL_FLOAT, 0);
        //glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA32F_ARB ,  matching_pixel_width, matching_pixel_height, 0, GL_ALPHA, GL_FLOAT, 0);

        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, matching_renderbuffer, 0);
        CheckOpenGlError("glFramebufferTexture2DEXT");

        status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);

        //make sure the framebuffer is complete
        if (status != GL_FRAMEBUFFER_COMPLETE_EXT)
        {
            SIMPLELOGGER_ERROR("Can't initialise OpenGL matching render buffer (%dx%d pixels).", matching_pixel_width, matching_pixel_height);
            SIMPLELOGGER_ERROR("All submap matching disabled.");
            glDeleteRenderbuffersEXT(1, &matching_renderbuffer);
            glDeleteFramebuffersEXT(1, &matching_framebuffer);
            matching_framebuffer = 0;
            matching_renderbuffer = 0;
            //exit(-1);
        }
        else
        {
            SIMPLELOGGER_INFO("Initialised matching render buffer (%dx%d pixels).", matching_pixel_width, matching_pixel_height);
        }

        SIMPLELOGGER_INFO("Compiling OpenGL GLSL shaders...");

        //Submap blending GLSL shaders
        //Simple vertex shader, pass both texture coordinate and local submap coordinate to fragment shader.
        const GLchar *submap_blend_vs_code = " \
				varying vec3 submap_coord;     \
				void main() {                \
					gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; \
					submap_coord = gl_Vertex.xyz;         \
					gl_TexCoord[0] = gl_MultiTexCoord0; \
				}";
        const GLchar *submap_blend_fs_code = "                                \
				varying vec3 submap_coord;         \
				uniform sampler2D submap_tex;    \
				void main() {                    \
					vec4 x = texture2D(submap_tex, gl_TexCoord[0].xy );         \
					float e = x.r-0.5;         \
					float e_factor = clamp(16.0*(e*e*e*e), 0.0, 1.0 );       \
					float r_factor = 0.1 + 0.02*dot(submap_coord.xy,submap_coord.xy);         \
					gl_FragDepth = e_factor * (clamp( -0.5*submap_coord.z, 0.0, 0.5) + clamp( 0.1/r_factor, 0.0, 0.5));   \
					gl_FragColor.r = x.r;        \
					gl_FragColor.g = x.r;        \
					gl_FragColor.b = gl_FragDepth;        \
				}";

        //gl_FragDepth = 1 is closest
        //gl_FragDepth = 0 is most distant
        //0m from origin, r_factor=0.1
        //5m from origin, r_factor=0.6
        //10m             r_factor=2.1
        //object, x.r=0.0, e=-0.5, e_factor=1.0
        //white,  x.r=1.0, e= 0.5, e_factor=1.0
        //grey,   x.r=0.5, e= 0.0, e_factor=0.0

        submap_blend_vs = glCreateShader(GL_VERTEX_SHADER);
        submap_blend_fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(submap_blend_vs, 1, &submap_blend_vs_code, 0);
        glCompileShader(submap_blend_vs);
        glShaderSource(submap_blend_fs, 1, &submap_blend_fs_code, 0);
        glCompileShader(submap_blend_fs);
        const int MAX_INFO_LOG_SIZE = 1025;
        GLchar infoLog[MAX_INFO_LOG_SIZE];
        glGetShaderInfoLog(submap_blend_vs, MAX_INFO_LOG_SIZE, 0, infoLog);
        SIMPLELOGGER_INFO("Submap blending vertex shader [%s]", infoLog);
        glGetShaderInfoLog(submap_blend_fs, MAX_INFO_LOG_SIZE, 0, infoLog);
        SIMPLELOGGER_INFO("Submap blending fragment shader [%s]", infoLog);
        submap_blend_sp = glCreateProgram();
        glAttachShader(submap_blend_sp, submap_blend_vs);
        glAttachShader(submap_blend_sp, submap_blend_fs);
        glLinkProgram(submap_blend_sp);
        glGetProgramInfoLog(submap_blend_sp, MAX_INFO_LOG_SIZE, 0, infoLog);
        SIMPLELOGGER_INFO("Submap blending program [%s]", infoLog);
        submap_blend_tex = glGetUniformLocation(submap_blend_sp, "submap_tex");

        /////////////submap matcher
        //experimental
        const GLchar *submap_matcher_vs_code = " \
				varying vec3 x_i;			 \
				void main() {                \
					gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; \
					x_i = gl_Vertex.xyz;         \
				}";

        const GLchar *submap_matcher_fs_code = "     \
				varying vec3 x_i;                    \
				uniform sampler2D base_submap_tex;    \
				uniform sampler2D target_submap_tex;     \
				uniform float scale;                 \
				uniform vec2 base_bottom_left;    \
				uniform vec2 base_top_right;      \
				uniform vec2 target_bottom_left;     \
				uniform vec2 target_top_right;       \
				void main() {                        \
					vec2 current_extents = (base_top_right-base_bottom_left);    \
					vec2 inv_target_extents = 1.0/(target_top_right-target_bottom_left) * vec2(1.0,-1.0);    \
					float gaussian_sum = 0.0;          \
					float cos_phi = cos(x_i.z);   \
					float sin_phi = sin(x_i.z);   \
					mat2 rot = mat2(cos_phi,-sin_phi,sin_phi,cos_phi);  \
					for (float y=base_bottom_left.y+0.05; y<=base_top_right.y; y+=0.2) {    \
						float yy = 1.0-(y-base_bottom_left.y)/current_extents.y;                 \
						float yyy = y-x_i.y; \
						for (float x=base_bottom_left.x+0.05; x<=base_top_right.x; x+=0.2) {    \
							vec4 current = texture2D( base_submap_tex,  vec2( (x-base_bottom_left.x)/current_extents.x, yy ) );    \
							vec4 target  = texture2D( target_submap_tex,((rot*vec2(x-x_i.x, yyy))-target_bottom_left) * inv_target_extents + vec2(0.0,1.0) ); \
							gaussian_sum += current.g*target.g;  \
						}                            \
					}                                \
					gl_FragColor.a = gaussian_sum;   \
				}";

        //TODO optimise target tex lookup
        //todo split channels O, G
        //			vec4 target  = texture2D( target_submap_tex,((rot*vec2(x-x_i.x, yyy))-target_bottom_left) * inv_target_extents + vec2(0.0,1.0) );
        //							vec4 target  = texture2D( target_submap_tex,((rot*vec2(x,y)-x_i.xy)-target_bottom_left) * inv_target_extents + vec2(0.0,1.0) );

        submap_matcher_vs = glCreateShader(GL_VERTEX_SHADER);
        submap_matcher_fs = glCreateShader(GL_FRAGMENT_SHADER);

        glShaderSource(submap_matcher_vs, 1, &submap_matcher_vs_code, 0);
        glCompileShader(submap_matcher_vs);
        glShaderSource(submap_matcher_fs, 1, &submap_matcher_fs_code, 0);
        glCompileShader(submap_matcher_fs);
        glGetShaderInfoLog(submap_matcher_vs, MAX_INFO_LOG_SIZE, 0, infoLog);
        SIMPLELOGGER_INFO("Submap matching vertex shader [%s]", infoLog);

        glGetShaderInfoLog(submap_matcher_fs, MAX_INFO_LOG_SIZE, 0, infoLog);
        SIMPLELOGGER_INFO("Submap matching fragment shader [%s]", infoLog);

        submap_matcher_sp = glCreateProgram();
        glAttachShader(submap_matcher_sp, submap_matcher_vs);
        glAttachShader(submap_matcher_sp, submap_matcher_fs);
        glLinkProgram(submap_matcher_sp);
        glGetProgramInfoLog(submap_matcher_sp, MAX_INFO_LOG_SIZE, 0, infoLog);
        SIMPLELOGGER_INFO("Submap matching program [%s]", infoLog);

        submap_matcher_scale = glGetUniformLocation(submap_matcher_sp, "scale");
        submap_matcher_base_bottom_left = glGetUniformLocation(submap_matcher_sp, "base_bottom_left");
        submap_matcher_base_top_right = glGetUniformLocation(submap_matcher_sp, "base_top_right");
        submap_matcher_target_bottom_left = glGetUniformLocation(submap_matcher_sp, "target_bottom_left");
        submap_matcher_target_top_right = glGetUniformLocation(submap_matcher_sp, "target_top_right");
        submap_matcher_initial_offset = glGetUniformLocation(submap_matcher_sp, "initial_offset");
        submap_matcher_base_submap_tex = glGetUniformLocation(submap_matcher_sp, "base_submap_tex");
        submap_matcher_target_submap_tex = glGetUniformLocation(submap_matcher_sp, "target_submap_tex");

        CheckOpenGlError("glGetUniformLocation");

        cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);
        cvInitFont(&font_small, CV_FONT_HERSHEY_SIMPLEX, 0.4, 0.4, 0, 1, CV_AA);
        f_encoder = 0;
        ground_truth_img = 0;
        plot_tracks = true;
        plot_constraints = false;

        matching_submap_iterator = map->submaps_list.rbegin();

        robot_color_rgb.push_back( cvScalar( 184,  44,  44) ); //red
        robot_color_rgb.push_back( cvScalar(  44, 184,  60) ); //green
        robot_color_rgb.push_back( cvScalar(  60,  44, 184) ); //blue
        robot_color_rgb.push_back( cvScalar( 184,  44, 156) ); //purple
        robot_color_rgb.push_back( cvScalar(  44, 163, 184) ); //cyan
        robot_color_rgb.push_back( cvScalar( 139,  82,   9) ); //brown
        robot_color_rgb.push_back( cvScalar( 143, 210,  41) ); //light green
        robot_color_rgb.push_back( cvScalar( 129,  41, 210) ); //dk prple
        robot_color_rgb.push_back( cvScalar( 210,  41, 120) ); //pink

        SIMPLELOGGER_INFO("Initialised.");
    }

    double MapBuilder::OptimiseLocalConstraintCost(SysSPA2d &spa, SubmapConstraint* test_constraint)
    {
        double cost = 0.0; //= spa.calcCost(false);
        int count = 0;
        int ndi0 = test_constraint->base_submap->optimisation_id;
        int ndi1 = test_constraint->target_submap->optimisation_id;

        //spa internal index to the test constraint's nodes
        int ni0 = -1, ni1 = -1;

        for (int i = 0; i < (int) spa.nodes.size(); i++)
        {
            if (spa.nodes[i].nodeId == ndi0)
                ni0 = i;
            if (spa.nodes[i].nodeId == ndi1)
                ni1 = i;
        }

        if (ni0 < 0 || ni1 < 0) return 9999999;

        for (size_t i = 0; i < spa.p2cons.size(); i++)
        {
            Con2dP2 &con = spa.p2cons[i];

            if ((con.nd1 == ni0) || (con.nd1 == ni1) || (con.ndr == ni0) || (con.ndr == ni1))
            {
                //this constraint links one of the test_constraint's nodes
                con.err.block < 2, 1 > (0, 0) = spa.nodes[con.ndr].w2n * spa.nodes[con.nd1].trans - con.tmean;
                double aerr = (spa.nodes[con.nd1].arot - spa.nodes[con.ndr].arot) - con.amean;
                if (aerr > M_PI) aerr -= 2.0 * M_PI;
                if (aerr < -M_PI) aerr += 2.0 * M_PI;
                con.err(2) = aerr;
                cost += con.err.dot(con.prec * con.err);
                count++;
                //cost += con.calcErrDist(spa.nodes[con.ndr],spa.nodes[con.nd1]);
            }
        }

        return cost / count;
    }

    bool MapBuilder::OptimiseSubmaps(SubmapConstraint* test_constraint)
    {
        //where the real work happens!
        //todo make more efficient by not duplicating node/constraint information
        //todo no ground truth unfix first submap
        //clean interface with SPA2d
        CTicTac tictac; //timer
        tictac.Tic();
        CTicTac setup_time;
        setup_time.Tic();

        //submap poses and constraints are (mostly) immutable
        //todo copy ONLY new poses and constraints into SPA
        SIMPLELOGGER_DEBUG("Optimisation: Setting up...");
        std::list<Submap*> free;

        //search for free nodes first
        BOOST_FOREACH(Submap* submap, map->GetSubmapList())
        {
            if (submap->IsFree() && submap->IsConstrained())
            {
                //this submap isn't fixed (by the GCS) and it has constraints acting on it
                free.push_back(submap);
            }
            submap->optimisation_id = 0; //reset the opt id
        }

        if (free.size() < 1)
        {
            SIMPLELOGGER_INFO("Optimisation: no free submap nodes, skipping optimisation.");
            return true; //true to accept any test constraints
        }

        SIMPLELOGGER_INFO("Optimisation: %d free submap nodes.", (int) free.size());

        stats["opt submap total count"] = map->GetSubmapList().size();
        stats["opt submap free count"] = free.size();

        //setup spa2d
        //main optimisation instance:
        SysSPA2d spa;
        int id = 100;
        int count_ground_truths = 0;
        int count_fixed = 0;
        int count_interpose_constraints = 0;

        //free nodes are constrained: ie. joined to the network
        //add the fixed ground-truth nodes
        BOOST_FOREACH(Submap* submap, free)
        {
            if (submap->GetGroundTruth())
            {
                SubmapGroundTruth* ground_truth = submap->GetGroundTruth();

                Vector3d pose(ground_truth->GetMean().x(), ground_truth->GetMean().y(), ground_truth->GetMean().phi());
                ground_truth->optimisation_id = id++;

                spa.addNode2d(pose, ground_truth->optimisation_id);
                count_ground_truths++;

                //todo better id's
            }
        }

        SIMPLELOGGER_INFO("Optimisation: %d ground truth constraints.", count_ground_truths);
        stats["opt ground truth constraint count"] = count_ground_truths;

        //add any fixed submaps that are linked to free ones by constraints
        BOOST_FOREACH(Submap* submap, free)
        {
            //first search constraints from this free node
            BOOST_FOREACH(SubmapConstraint* constraint, submap->GetBaseConstraints())
            {
                Submap* target_submap = constraint->target_submap;

                if ((target_submap->IsFixed()) //todo possibly bad pose?
                        && (target_submap->optimisation_id == 0))
                {
                    //this constraint target is fixed, but not added yet:
                    CPose2D _pose = target_submap->GetPose()->GetMean();
                    Vector3d pose(_pose.x(), _pose.y(), _pose.phi());
                    target_submap->optimisation_id = id++;
                    spa.addNode2d(pose, target_submap->optimisation_id);
                    count_fixed++;
                }
            }
            //then any constraints pointing to this free node

            BOOST_FOREACH(SubmapConstraint* constraint, submap->GetTargetConstraints())
            {
                Submap* base_submap = constraint->base_submap;

                if ((base_submap->IsFixed()) //todo possibly bad pose?
                        && (base_submap->optimisation_id == 0))
                {
                    //this constraint base is fixed, but not added yet:
                    CPose2D _pose = base_submap->GetPose()->GetMean();
                    Vector3d pose(_pose.x(), _pose.y(), _pose.phi());
                    base_submap->optimisation_id = id++;
                    spa.addNode2d(pose, base_submap->optimisation_id);
                    count_fixed++;
                }
            }
        }

        SIMPLELOGGER_INFO("Optimisation: %d fixed submaps.", count_fixed);
        stats["opt submap fixed count"] = count_fixed;

        spa.nFixed = count_fixed + count_ground_truths;

        if (spa.nFixed < 1)
        {
            spa.nFixed = 1;
            SIMPLELOGGER_WARN("Optimisation: no fixed nodes found.");
        }

        //now add all the free nodes

        BOOST_FOREACH(Submap* submap, free)
        {
            CPose2D _pose; //set to zeros

            if (submap->GetPose())
                _pose = submap->GetPose()->GetMean();
            else
                SIMPLELOGGER_WARN("Optimisation: submap %s has (0,0,0) initial pose.", submap->GetUUID().substr(0, 6).c_str());

            //TODO nonoogoonononooooooooooooooo bad bad bad
            Vector3d pose(_pose.x(), _pose.y(), _pose.phi());

            if (submap->optimisation_id) SIMPLELOGGER_ERROR("Optimisation: submap already has opt id %d", submap->optimisation_id);

            submap->optimisation_id = id++;
            spa.addNode2d(pose, submap->optimisation_id);
            //todo better id's
        }

        //now add the constraints:
        //add ground truth constraints
        BOOST_FOREACH(Submap* submap, free)
        {
            if (submap->GetGroundTruth())
            {
                SubmapGroundTruth* ground_truth = submap->GetGroundTruth();
                Vector3d mean(0.0, 0.0, 0.0);
                spa.addConstraint2d(ground_truth->optimisation_id,
                        submap->optimisation_id,
                        mean,
                        ground_truth->precision);
            }
        }

        //now add inter-submap constraints
        BOOST_FOREACH(Submap* submap, free)
        {
            //only add base constraints, the "other" submap will add when this is target
            BOOST_FOREACH(SubmapConstraint* constraint, submap->GetBaseConstraints())
            {
                Vector3d mean(constraint->constraint_mean.x(),
                        constraint->constraint_mean.y(),
                        constraint->constraint_mean.phi());
                spa.addConstraint2d(constraint->base_submap->optimisation_id,
                        constraint->target_submap->optimisation_id,
                        mean,
                        constraint->constraint_precision);
                //todo better id's
                count_interpose_constraints++;
            }

            //add any constraints pointing from a fixed node to this free node
            BOOST_FOREACH(SubmapConstraint* constraint, submap->GetTargetConstraints())
            {
                Submap* base_submap = constraint->base_submap;
                if (base_submap->IsFixed())
                {
                    //this constraint's base is a fixed node, add it
                    Vector3d mean(constraint->constraint_mean.x(),
                            constraint->constraint_mean.y(),
                            constraint->constraint_mean.phi());
                    spa.addConstraint2d(constraint->base_submap->optimisation_id,
                            constraint->target_submap->optimisation_id,
                            mean,
                            constraint->constraint_precision);
                    count_interpose_constraints++;
                }
            }
        }

        SIMPLELOGGER_INFO("Optimisation: %d inter-submap constraints.", count_interpose_constraints);
        stats["opt inter submap constraint count"] = count_interpose_constraints;
        stats["opt setup time"] = setup_time.Tac();
        CTicTac opt_time;
        opt_time.Tic();

        ///////////////////////////////////////////////////
        int iterations = spa.doSPA(10);
        ///////////////////////////////////////////////////

        SIMPLELOGGER_INFO("Optimisation: SPA finished in %d iterations, %1.1f msec.", iterations, 1000.0 * tictac.Tac());

        stats["opt opt time"] = opt_time.Tac();
        stats["opt opt iterations"] = iterations;
        stats["opt constraint cost rms"] = sqrt(spa.calcCost() / spa.p2cons.size());
        stats["opt graph vertices count"] = spa.nodes.size();
        stats["opt graph edges count"] = spa.p2cons.size();

        if (test_constraint)
        {
            CTicTac test_time;
            test_time.Tic();

            //we're just testing a constraint
            double cost_before = OptimiseLocalConstraintCost(spa, test_constraint);

            Vector3d mean(test_constraint->constraint_mean.x(), test_constraint->constraint_mean.y(), test_constraint->constraint_mean.phi());
            spa.addConstraint2d(test_constraint->base_submap->optimisation_id,
                    test_constraint->target_submap->optimisation_id, mean, test_constraint->constraint_precision);

            tictac.Tic(); //redo opt with test constraint
            int iterations = spa.doSPA(10);

            double cost_after = OptimiseLocalConstraintCost(spa, test_constraint); //spa.calcCost(false);
            SIMPLELOGGER_INFO("Optimisation: SPA finished in %d iterations, %1.1f msec.", iterations, 1000.0 * tictac.Tac());
            SIMPLELOGGER_INFO("Optimisation: cost before test_constraint %1.6f", cost_before);
            SIMPLELOGGER_INFO("Optimisation: cost with   test_constraint %1.6f", cost_after);

            stats["match constraint test time"] += test_time.Tac();
            stats["match constraint test count"]++;
            return (cost_after < (cost_before * (1.0+COST_INCREASE_THRESHOLD) + COST_MINIMUM_THRESHOLD)); //+0.05
        }

        //get the results:
        std::vector<sba::Node2d, Eigen::aligned_allocator<sba::Node2d> > nodes = spa.getNodes();
        std::list<Submap*> updated;
        std::list<Submap*> send;

        BOOST_FOREACH(sba::Node2d node, nodes)
        {
            if (node.isFixed) continue; //skip if this node was fixed

            Submap* submap = 0;

            //search for the submap with this opt id
            //todo better indexing ~horribly inefficient
            BOOST_FOREACH(Submap* s, free) if (s->optimisation_id == node.nodeId)
            {
                submap = s;
                break;
            }

            if (!submap)
            {
                SIMPLELOGGER_ERROR("Optimisation: submap opt id %d not found.", node.nodeId);
                continue;
            }

#define SUBMAP_POSE_CORRECTION_MAX_LIN_DIST 2000

            //submap is now valid, updated submap.
            //stats["opt submap pose update count"]=0;
            CPose2D pose(node.trans(0), node.trans(1), node.arot);
            if (submap->GetPose())
            {
                CPose2D delta;
                if (submap->last_sent_pose)
                    delta = pose - *submap->last_sent_pose;
                else
                    delta = pose - submap->GetPose()->GetMean();

                if (delta.norm() < SUBMAP_POSE_CORRECTION_MAX_LIN_DIST)
                {
                    //ignore large jumps
                    submap->GetPose()->SetMean(pose); //update the actual pose

                    //only send the pose if it changes by more than a threshold
                    if ((delta.norm() > SUBMAP_POSE_CORRECTION_MIN_LIN_DIST)
                            || (fabs(delta.phi()) > SUBMAP_POSE_CORRECTION_MIN_ANG_DIST))
                    {
                        //very sloppy tolerances to prevent pose thrashing
                        updated.push_back(submap); //resend this pose
                        map->graph_changed = true;
                        stats["opt submap pose update count"]++;
                    }
                }
            }
            else
            {
                SIMPLELOGGER_DEBUG("Optimisation: %s_ now has a pose", submap->GetUUID().substr(0, 6).c_str());
                CMatrixDouble33 cov;
                cov.eye();
                SubmapPose* sp = new SubmapPose(submap->GetUUID(), pose, cov, false); //todo fix cov
                submap->SetPose(sp);
                updated.push_back(submap);
                stats["opt submap pose update count"]++;
            }
        }

        SIMPLELOGGER_DEBUG("Optimisation: updated %d submap poses.", (int) updated.size());

        if (!is_GCS)
        {
            //if this is a robot with a current_submap, send it if its been updated
            //otherwise ignore the rest of the updated submaps: we dont send them
            BOOST_FOREACH(Submap* submap, updated)
            if (submap == current_submap)
                send.push_back(submap);
        }
        else
        {
            //this is the gcs, send all updated poses
            BOOST_FOREACH(Submap* submap, updated)
            send.push_back(submap);
        }

        SIMPLELOGGER_DEBUG("Optimisation: finished.");
        return true;
    }

#define SUBMAP_MAX_RADIUS 15.0

    bool MapBuilder::BuildMap(int dt_rel_last, bool save_frame)
    {
        double ts_now = (double)dt_rel_last;// * TS_SEC; //now();
        SIMPLELOGGER_DEBUG("Build() started...");

        //first quickly pump hidden windows message queue
        /*		MSG	msg;
                        while( PeekMessage( &msg, hWnd, 0, 0, PM_REMOVE )) {
                                TranslateMessage( &msg );
                                DispatchMessage( &msg );
                        }
         */
        //loop over submaps to find global map extents
        //submaps could be at   0,0 in Grid zero,
        //						0,0 in current grid, or
        //						x,y in current grid
        //build the map around current_submap, GCS may have issues here TODO
        CPose2D *map_reference_pose = new CPose2D();
        //		if (robot_pose) {
        //			//use the current submap
        //			map_reference_pose = (CPose2D*)&current_submap->GetPose()->GetMean();
        //		} else {
        //			//todo rr nov 6th
        //			map_reference_pose = &reference_position;
        //otherwise, use the latest submap's pose
        //BOOST_FOREACH(Submap* submap in map->GetSubmapList() )
        //	if (submap->GetPose()) map_reference_pose = (CPose2D*)&submap->GetPose()->GetMean();
        //		}

        bool can_map = false;
        double x_min = 1e99;
        double y_min = 1e99;
        double x_max = -1e99;
        double y_max = -1e99;
        double ts_min = ts_now - 5;
        double ts_max = ts_now;

        BOOST_FOREACH(Submap* submap, map->GetSubmapList())
        {
            //build submap grid data buffers if new tiles waiting
            //this also writes update submaps to PNG/disk
            submap->BuildSubmapData();

            if (submap->GetPose() && submap->GetOccupancyData())
            {
                CPose2D pose = submap->GetPose()->GetMean();
                CPoint2D tl = pose + submap->GetExtentsTopLeft();
                CPoint2D br = pose + submap->GetExtentsBottomRight();
                CPoint2D tr = pose + CPoint2D( submap->GetExtentsBottomRight().x(), submap->GetExtentsTopLeft().y() );
                CPoint2D bl = pose + CPoint2D( submap->GetExtentsTopLeft().x(), submap->GetExtentsBottomRight().y() );

                //				//if (map_reference_pose->distanceTo( pose ) < 500.0) {
                //				//if this submap is within 500m of the current submap, consider it for mapping
                can_map = true;
                x_min = min(x_min, tl.x() );
                y_min = min(y_min, tl.y() );
                x_max = max(x_max, tl.x() );
                y_max = max(y_max, tl.y() );
                x_min = min(x_min, tr.x() );
                y_min = min(y_min, tr.y() );
                x_max = max(x_max, tr.x() );
                y_max = max(y_max, tr.y() );
                x_min = min(x_min, bl.x() );
                y_min = min(y_min, bl.y() );
                x_max = max(x_max, bl.x() );
                y_max = max(y_max, bl.y() );
                x_min = min(x_min, br.x() );
                y_min = min(y_min, br.y() );
                x_max = max(x_max, br.x() );
                y_max = max(y_max, br.y() );
                x_min = min(x_min, pose.x()-SUBMAP_MAX_RADIUS);
                y_min = min(y_min, pose.y()-SUBMAP_MAX_RADIUS);
                x_max = max(x_max, pose.x()+SUBMAP_MAX_RADIUS);
                y_max = max(y_max, pose.y()+SUBMAP_MAX_RADIUS);

            }
            ts_min = min(ts_min, submap->GetTimeStamp());
            ts_max = max(ts_max, submap->GetTimeStamp());
        }

        if (can_map == false)
        {
            SIMPLELOGGER_INFO("No data to map.");
            return false;
        }

        //Upper Left    (  279301.412, 6130556.050)  (138.5833379,-34.9434970)

        //build map:
        //round to 256 pixel boundaries for HMI
        x_min = floor( x_min*OccupancyTile::CELLS_PER_M/generatedMapSizeMultiple)*generatedMapSizeMultiple/OccupancyTile::CELLS_PER_M;
        y_min = floor( y_min*OccupancyTile::CELLS_PER_M/generatedMapSizeMultiple)*generatedMapSizeMultiple/OccupancyTile::CELLS_PER_M;
        x_max = ceil(  x_max*OccupancyTile::CELLS_PER_M/generatedMapSizeMultiple)*generatedMapSizeMultiple/OccupancyTile::CELLS_PER_M;
        y_max = ceil(  y_max*OccupancyTile::CELLS_PER_M/generatedMapSizeMultiple)*generatedMapSizeMultiple/OccupancyTile::CELLS_PER_M;

        //todo crop max map size
        //todo nov 6th
        //sliding window:
        double max_width  = (double)map_pixel_width /OccupancyTile::CELLS_PER_M;
        double max_height = (double)map_pixel_height/OccupancyTile::CELLS_PER_M;

        if ((x_max-x_min) > max_width )
        {
                //set width around robot pose
                double x_min_new = map_reference_pose->x() - max_width/2.0;
                double x_max_new = map_reference_pose->x() + max_width/2.0;

                if (x_max_new > x_max)
                { //expanded, move extra to other edge
                        double inc = x_max_new - x_max;
                        x_max_new -= inc;
                        x_min_new -= inc;
                }

                if (x_min_new < x_min)
                { //expanded, move extra to other edge
                        double inc = x_min - x_min_new;
                        x_min_new += inc;
                        x_max_new += inc;
                }

                x_min = x_min_new;
                x_max = x_max_new;
        }

        if ((y_max-y_min) > max_height )
        {
                //set around robot pose
                double y_min_new = map_reference_pose->y() - max_height/2.0;
                double y_max_new = map_reference_pose->y() + max_height/2.0;

                if (y_max_new > y_max)
                { //expanded, move extra to other edge
                        double inc = y_max_new - y_max;
                        y_max_new -= inc;
                        y_min_new -= inc;
                }

                if (y_min_new < y_min)
                { //expanded, move extra to other edge
                        double inc = y_min - y_min_new;
                        y_min_new += inc;
                        y_max_new += inc;
                }

                y_min = y_min_new;
                y_max = y_max_new;
        }

        SIMPLELOGGER_INFO("Map extents, width: %1.2f, height: %1.2f, top left: (%1.2f,%1.2f) bottom right: (%1.2f,%1.2f)", x_max - x_min, y_max - y_min, x_min, y_max, x_max, y_min);

        this->x_max = x_max;
        this->x_min = x_min;
        this->y_max = y_max;
        this->y_min = y_min;

        stats["build map width"] = x_max - x_min;
        stats["build map height"] = y_max - y_min;

        CTicTac tictac;
        tictac.Tic();
        CTicTac setup_time;
        setup_time.Tic();
        int count = 0;

        CheckOpenGlError("Error");

        //bind map framebuffer, setup ortho projection, clear buffer and load GLSL shader
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, map_framebuffer); //Bind to it
        CheckOpenGlError("glBindFramebufferEXT map_framebuffer");

        //glPushAttrib(GL_VIEWPORT_BIT);
        glViewport(0, 0, map_pixel_width, map_pixel_height);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, (float) map_pixel_width / (float) OccupancyTile::CELLS_PER_M,
                0, (float) map_pixel_height / (float) OccupancyTile::CELLS_PER_M,
                0.0f, 1.0f);

        glShadeModel(GL_SMOOTH);
        glDisable(GL_BLEND);
        glClearColor(0.47f, 0.47f, 0.0f, 1.0f); //0.47f, 0.47f, 0.0f, 1.0f
        glClearDepth(0.0f);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_GEQUAL); //1.0 is closest depth
        glMatrixMode(GL_MODELVIEW);
        glActiveTexture(GL_TEXTURE0);

        //		SIMPLELOGGER_INFO( "glClear start" );

        //stupid hack for Intel GPU,
        //		glDisable(GL_TEXTURE_2D); //clear using screen-aligned quad, only current map area
        //		glDepthFunc(GL_LEQUAL); //invert for clear
        //		glLoadIdentity();
        //		glTranslatef( -x_min, -y_min, 0.0);
        //		glColor4f(0.5f,0.0f,0.0f,1.0f);
        //		glBegin(GL_QUADS);
        //			glVertex3f( x_min, y_max, 0.0 );//Top left
        //			glVertex3f( x_max, y_max, 0.0 );//Top right
        //			glVertex3f( x_max, y_min, 0.0 );//Bottom right
        //			glVertex3f( x_min, y_min, 0.0 );//Bottom left
        //		glEnd();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        CheckOpenGlError("glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)");
        //SIMPLELOGGER_INFO( "glClear end" );

        glUseProgram(submap_blend_sp);
        CheckOpenGlError("glUseProgram Submap blending program");
        //glFinish();

        stats["build setup time"] = setup_time.Tac();

        //submaps stack from Z=0.0 (oldest) to Z=1.0 (newest)
        ts_min -= 1;
        ts_max += 1;
        double ts_range = ts_max - ts_min;
        std::list<Submap*> submaps_to_plot;
        CTicTac texture_time;
        texture_time.Tic();

        //loop over submap
        stats["build submap occupied cell count"] = 0;
        stats["build submap free cell count"]     = 0;
        stats["build submap unknown cell count"]  = 0;

        BOOST_FOREACH(Submap* submap, map->GetSubmapList())
        {
            //if the submap doesn't have a pose, skip it
            if (!submap->GetPose()) continue;
            CPose2D pose = submap->GetPose()->GetMean();

            //if the submap doesn't have any occupancy data skip it
            IplImage* occupancy_img = submap->GetOccupancyData();
            if (!occupancy_img) continue;

            if (  !(((pose.x() + SUBMAP_MAX_RADIUS) > x_min) &&
                    ((pose.x() - SUBMAP_MAX_RADIUS) < x_max) &&
                    ((pose.y() + SUBMAP_MAX_RADIUS) > y_min) &&
                    ((pose.y() - SUBMAP_MAX_RADIUS) < y_max)))
            {
                //if submap outside window
                GLuint *texture = occupancy_texture[submap->GetUUID()];
                if (texture != NULL)
                {
                    //free the old texture
                    glDeleteTextures(1, texture);
                    CheckOpenGlError("glDeleteTextures out of window");
                    delete texture;
                    occupancy_texture[submap->GetUUID()] = NULL;
                    stats["build texture unloaded count"]++;
                    SIMPLELOGGER_DEBUG("Unloaded texture for submap %s_", submap->GetUUID().substr(0, 6).c_str());
                }
                continue;
            }

            //we've got what we need to blit this submap:
            count++;
            submaps_to_plot.push_back(submap);
            stats["build submap occupied cell count"] += submap->GetCellCountOccupied();
            stats["build submap free cell count"] += submap->GetCellCountFree();
            stats["build submap unknown cell count"] += submap->GetCellCountUnknown();

            //keeping all opengl code here:
            //lookup map for current texture
            if (submap->GetOccupancyUpdatedFlag())
            {
                //this submap's data has been changed, refresh
                //TODO dont delete if its the same size: save GPU mem reallocation
                GLuint *texture = occupancy_texture[submap->GetUUID()];
                if (texture)
                {
                    //free the old texture
                    glDeleteTextures(1, texture);
                }
                else
                {
                    //create new texture
                    texture = new GLuint;
                }

                glGenTextures(1, texture);
                CheckOpenGlError("glGenTextures updating submap tex");
                glBindTexture(GL_TEXTURE_2D, *texture);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); //GL_LINEAR
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); //GL_NEAREST
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

                /* Colour the submap based on the robot that made it
                    IplImage *overlay = cvCreateImage(cvGetSize(occupancy_img), IPL_DEPTH_8U, 3);
                    IplImage *final_image = cvCreateImage(cvGetSize(occupancy_img), IPL_DEPTH_8U, 3);

                    cvCopy(occupancy_img, overlay);

                    cv::Mat image1(overlay);
                    cv::Mat image2(occupancy_img);
                    cv::Mat image3(final_image);

                    CvScalar colour = robot_color_rgb[submap->robot_id];
                    cv::rectangle(image1, cv::Point(0, 0), cv::Point(occupancy_img->width, occupancy_img->height), colour, -1, 8, 0);

                    cv::addWeighted(image1, 0.4, image2, 1 - 0.4, 0, image3);

                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, final_image->width, final_image->height, 0, GL_RGB, GL_UNSIGNED_BYTE, image3.data);

                    //dump submap tile to disk for persistence
                    char submap_filename[255];
                    sprintf(submap_filename, "%s/submaps/%s.png", DATA_PATH, submap->GetUUID().c_str() );
                    cvSaveImage(submap_filename, final_image);

                    cvReleaseImage(&final_image);
                    cvReleaseImage(&overlay);
                */

                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, occupancy_img->width, occupancy_img->height, 0, GL_RGB, GL_UNSIGNED_BYTE, occupancy_img->imageData);
                submap->ClearOccupancyUpdatedFlag();
                occupancy_texture[submap->GetUUID()] = texture;
                //SIMPLELOGGER_DEBUG( "Reloaded texture for submap %s_", submap->GetUUID().substr(0,6).c_str() );
                stats["build texture loaded count"]++;
            }
        }//end loop over submaps loading textures

        glFinish();
        stats["build texture time"] += texture_time.Tac();
        CTicTac render_time;
        render_time.Tic();

        //loop over submaps we can plot
        BOOST_FOREACH(Submap* submap, submaps_to_plot)
        {
            GLuint *texture = occupancy_texture[submap->GetUUID()];

            if (!texture) continue;

            CPose2D pose = submap->GetPose()->GetMean();
            //earliest ts depth is 0.0, newest is 1.0
            GLfloat depth = (submap->GetTimeStamp() - ts_min) / ts_range;

            glBindTexture(GL_TEXTURE_2D, *texture);
            CheckOpenGlError("glBindTexture for submap occ data");
            glUniform1i(submap_blend_tex, 0);

            GLfloat x_left = submap->GetExtentsTopLeft().x();
            GLfloat y_top = submap->GetExtentsTopLeft().y();
            GLfloat x_right = submap->GetExtentsBottomRight().x();
            GLfloat y_bottom = submap->GetExtentsBottomRight().y();

            glLoadIdentity();
            glTranslatef(pose.x() - x_min, pose.y() - y_min, 0.0);
            glRotatef(R2D(pose.phi()), 0.0f, 0.0f, 1.0f);

            glBegin(GL_QUADS); //texture is stored row0-first = upside down for opengl
                glTexCoord2f(0.0f, 0.0f); glVertex3f(x_left,  y_top,    -depth); //Top left
                glTexCoord2f(1.0f, 0.0f); glVertex3f(x_right, y_top,    -depth); //Top right
                glTexCoord2f(1.0f, 1.0f); glVertex3f(x_right, y_bottom, -depth); //Bottom right
                glTexCoord2f(0.0f, 1.0f); glVertex3f(x_left,  y_bottom, -depth); //Bottom left
            glEnd();
            //CheckOpenGlError("glEnd for submap occ data");
        }
        //end submap loop
        glFinish();
        stats["build render gpu time"] = render_time.Tac();
        stats["build submap render count"] = submaps_to_plot.size();
        stats["build submap total count"] = map->GetSubmapList().size();


        SIMPLELOGGER_INFO("Blitted %d Submaps average %1.3f ms each.", count, tictac.Tac()*1000.0f / count);

#if 0
        //plot submap constraints for debug
        if (is_GCS)
        {
            //plot debug submap constraints on the HMI only
            glDisable(GL_TEXTURE_2D);
            glDisable(GL_DEPTH_TEST);
            glLineWidth(1.0);
            glLoadIdentity();
            glTranslatef(-x_min, -y_min, 0.0);
            glColor3f(0.84f, 0.17f, 0.17f);
#endif
#if 0
            // draw constraints
            glBegin(GL_LINES);

            BOOST_FOREACH(Submap* submap, map->GetSubmapList())
            {
                if (submap->GetPose() && submap->GetOccupancyData() && submap->GetBaseConstraints().size() > 0)
                {

                    BOOST_FOREACH(SubmapConstraint* constraint, submap->GetBaseConstraints())
                    {
                        if (constraint->target_submap->GetOccupancyData())
                        {
                            //only draw base constraints, the "other" submap will be drawn when this is target
                            CPose2D target_pose = constraint->target_submap->GetPose()->GetMean();
                            CPose2D base_pose = submap->GetPose()->GetMean();
                            glVertex2f(target_pose.x(), target_pose.y());
                            glVertex2f(base_pose.x(), base_pose.y());
                        }
                    }
                }
            }
            glEnd();
            //#if 0
            //else
            glPointSize(1.0);
            glBegin(GL_POINTS);

            BOOST_FOREACH(Submap* submap, map->GetSubmapList())
            {
                //if ( submap->GetOccupancyData() ) {
                CPose2D pose = submap->GetPose()->GetMean();
                glVertex3f(pose.x(), pose.y(), 0.0);
                //}
            }
            glEnd();
#endif

            SIMPLELOGGER_DEBUG("Reading render buffer from GPU...");

            int w = (int) ((x_max - x_min) * OccupancyTile::CELLS_PER_M + 0.5); //round up
            int h = (int) ((y_max - y_min) * OccupancyTile::CELLS_PER_M + 0.5);

            map_img = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 3);
            composite_img = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 3);

            CTicTac read_time;
            read_time.Tic();
            glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, map_img->imageData);
            //unbind framebuffer
            //glPopAttrib();
            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
            CheckOpenGlError("glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0) build");
            stats["build map buffer read time"] = read_time.Tac();

            ////////////////////////////
            //calum send map_img as ros grid
            //nav_msgs::OccupancyGrid temp_map = new nav_msgs::OccupancyGrid();
            //ros_map(new nav_msgs::OccupancyGrid());
            std::string map_frame_ = "map";
            ros_map->header.frame_id =  map_frame_;
            ros_map->info.width = w;
            ros_map->info.height = h;

            //ROS set the resolution of the ros output map as 0.1 = 10cm
            ros_map->info.resolution = 1.0/OccupancyTile::CELLS_PER_M;
            ros_map->header.stamp = ros::Time::now();
            ros_map->info.origin.position.x = x_min;
            ros_map->info.origin.position.y = y_min;
            ros_map->data.resize(ros_map->info.width*ros_map->info.height);
            //fill(ros_map.data.begin() , ros_map.data.end(), -1);
            int free, occupied, unknown;
            free = occupied = unknown = 0;
            //ROS_INFO("w = %i, h= %i",w,h);
            typedef uint32_t index_t;
            typedef int16_t coord_t;
            coord_t map_x;
            coord_t map_y;
            index_t ros_map_pos;

            for (int y = 0; y < h; y++)
            {
                uchar *comp = (uchar *)&composite_img->imageData[map_img->widthStep * y];
                uchar *map = (uchar *)&map_img->imageData[map_img->widthStep * y];

                for (int x = 0; x < w; x++)
                {
                    int xtimes3 = x * 3;
                    map_x = x;
                    map_y = y;
                    ros_map_pos = map_x+map_y*w;

                    if (map[xtimes3 + 1] > 127)
                    {
                            free++;
                            ros_map->data[ros_map_pos] = (signed char) 0;

                    }
                    else if (map[xtimes3 + 1] < 120)
                    {
                        occupied++;
                        ros_map->data[ros_map_pos] = (signed char) 100;

                    }
                    else
                    {
                        unknown++;
                        ros_map->data[ros_map_pos] = (signed char) -1;

                    }
                }
            }

            ROS_INFO("unknown = %i, occupied= %i, free= %i",unknown,occupied,free);

            ////////////////////////////


            //SIMPLELOGGER_DEBUG("Done. Saving file...");
            //int p[3]; p[0] = CV_IMWRITE_JPEG_QUALITY; p[1] = 99; p[2] = 0; //not good, JPG uses YCbCr color space
            string map_filepath = format("%s/%s.%s", maps_path, generated_map_uuid.c_str(), GENERATED_MAP_FORMAT.c_str());
            cvConvertImage(map_img, map_img, CV_CVTIMG_FLIP); //|CV_CVTIMG_SWAP_RB );
            //CvRect r = cvRect( 0, 40, map_pixel_width-20, map_pixel_height-40 );
            //cvSetImageROI(map_img,r);

            //blend images
            //cvAddWeighted(map_img, 0.7, ground_truth_img, 0.3, 0.0, composite_img);
            cvCopy(map_img, composite_img);

			char temp_str[20];

            if (plot_tracks)
            {
                //plot robot positions
                for (int i = 0; i < MAX_ROBOTS; i++)
                {
                    if (robot_pose_history[i].size() > 1)
                    {
                        CPose2D pp = robot_pose_history[i].front().submap->GetPose()->GetMean() + robot_pose_history[i].front().pose;


                        CvPoint p = cvPoint((pp.x() - x_min)*OccupancyTile::CELLS_PER_M, (y_max - pp.y())*OccupancyTile::CELLS_PER_M);
                        CvPoint q = p;
                        CvScalar col = robot_color_rgb[i%robot_color_rgb.size()];
                        BOOST_FOREACH(RobotPoseLocal r, robot_pose_history[i])
                        {
                            pp = r.submap->GetPose()->GetMean() + r.pose;
                            p = cvPoint((pp.x() - x_min)*OccupancyTile::CELLS_PER_M, (y_max - pp.y())*OccupancyTile::CELLS_PER_M);
                            cvLine(composite_img, p, q, col);
                            q = p;
                        }
                    }
                }
            }

            if (plot_constraints)
            {
                CvScalar col  = cvScalar(225, 47, 57); //blue constraint lines
                CvScalar col1 = cvScalar(40, 172, 43); //green = submap origin
                CvScalar col2 = cvScalar(35, 35, 227); //red = has ground truth

                //plot constraints
                BOOST_FOREACH(Submap* submap, map->GetSubmapList())
                {
                    if (submap->GetPose() && submap->GetOccupancyData() && submap->GetBaseConstraints().size() > 0)
                    {
                        BOOST_FOREACH(SubmapConstraint* constraint, submap->GetBaseConstraints())
                        {
                            if (constraint->target_submap->GetOccupancyData())
                            {
                                //only draw base constraints, the "other" submap will be drawn when this is target
                                CPose2D target_pose = constraint->target_submap->GetPose()->GetMean();
                                CPose2D base_pose = submap->GetPose()->GetMean();
                                CvPoint p = cvPoint((target_pose.x() - x_min)*OccupancyTile::CELLS_PER_M, (y_max - target_pose.y())*OccupancyTile::CELLS_PER_M);
                                CvPoint q = cvPoint((base_pose.x() - x_min)*OccupancyTile::CELLS_PER_M, (y_max - base_pose.y())*OccupancyTile::CELLS_PER_M);
                                cvLine(composite_img, p, q, col);
                            }
                        }
                    }
                }
                //plot submap origin dots
                BOOST_FOREACH(Submap* submap, map->GetSubmapList())
                {
                    if (submap->GetPose() && submap->GetOccupancyData())
                    {
                        CPose2D pose = submap->GetPose()->GetMean();
                        CvPoint p = cvPoint((pose.x() - x_min)*OccupancyTile::CELLS_PER_M, (y_max - pose.y())*OccupancyTile::CELLS_PER_M);
                        //node pose green dot
                        cvCircle(composite_img, p, 3, col1, -1);
                    }
                }

                BOOST_FOREACH(Submap* submap, map->GetSubmapList())
                {
                    if (submap->GetPose() && submap->GetOccupancyData() && submap->GetGroundTruth())
                    {
                        CPose2D pose = submap->GetPose()->GetMean();
                        CvPoint p = cvPoint((pose.x() - x_min)*OccupancyTile::CELLS_PER_M, (y_max - pose.y())*OccupancyTile::CELLS_PER_M);
                        CPose2D gt = submap->GetGroundTruth()->GetMean();
                        int x=(gt.x() - x_min)*OccupancyTile::CELLS_PER_M;
                        int y=(y_max - gt.y())*OccupancyTile::CELLS_PER_M;
                        cvLine(composite_img, p, cvPoint(x,y), col2);
                        //red triangle
                        CvPoint t[3] = { cvPoint(x-4,y),cvPoint(x+4,y),cvPoint(x,y+7)};
                        cvFillConvexPoly (composite_img, t, 3, col2);
                        //cvCircle( composite_img, g, 4, col2, -1);
                    }
                }
            }

            //plot robot position marker last
            for (int i = MAX_ROBOTS-1; i >= 0; i--)
            {
                CvScalar col_white = cvScalar(255, 255, 255);

                if (robot_pose_history[i].size() > 1)
                {
                    CvScalar col = robot_color_rgb[i%robot_color_rgb.size()];
                    CPose2D pp = robot_pose_history[i].back().submap->GetPose()->GetMean() + robot_pose_history[i].back().pose;
                    CvPoint p = cvPoint((pp.x() - x_min)*OccupancyTile::CELLS_PER_M, (y_max - pp.y())*OccupancyTile::CELLS_PER_M);
                    CPose2D ppp = pp + CPoint2D(1.6,0);
                    CvPoint pppp = cvPoint((ppp.x() - x_min)*OccupancyTile::CELLS_PER_M, (y_max - ppp.y())*OccupancyTile::CELLS_PER_M);
                    cvLine(composite_img, p, pppp, col_white,2);
                    cvCircle(composite_img, p, 8, col_white, 1);
                    cvLine(composite_img, p, pppp, col);
                    cvCircle(composite_img, p, 7, col, -1);
                    sprintf(temp_str, "%c", i<10 ? (char)i+48 : (char)i+65-10-1);
                    cvPutText(composite_img, temp_str, cvPoint(p.x - 4, p.y + 4), &font_small, col_white);
                }
            }

            //plot submap matching scores for the most recent submap
            if (!submap_scores.empty())
            {
                CvScalar col_white = cvScalar(255, 255, 255);
                CvPoint startPoint = cvPoint(10, 20);

                int count = 0;
                double pair[2];
                char temp[50];

                for (std::list<double>::const_iterator it = submap_scores.begin(); it != submap_scores.end(); ++it)
                {
                    if (count < 2)
                    {
                        pair[count] = *it;
                        count++;
                    }
                    else
                    {
                        sprintf(temp, "Score %1.2f, Threshold %1.2f", pair[0], pair[1]);

                        cvPutText(composite_img, temp, startPoint, &font_small, col_white);

                        startPoint.y += 12;

                        count = 0;
                    }
                }

                if (submap_scores.size() > 40)
                {
                    submap_scores.clear();
                }
            }

            //plot cursor
            if (submap_highlight && submap_highlight->GetGroundTruth())
            {
                CPose2D gt = submap_highlight->GetGroundTruth()->GetMean();
                CPose2D gtx = gt + CPoint2D(OccupancyTile::CELLS_PER_M,0);
                CPose2D gty = gt + CPoint2D(0,OccupancyTile::CELLS_PER_M);
                CvPoint g  = cvPoint((gt.x() - x_min)*OccupancyTile::CELLS_PER_M, (y_max - gt.y())*OccupancyTile::CELLS_PER_M);
                CvPoint gx = cvPoint((gtx.x() - x_min)*OccupancyTile::CELLS_PER_M, (y_max - gtx.y())*OccupancyTile::CELLS_PER_M);
                CvPoint gy = cvPoint((gty.x() - x_min)*OccupancyTile::CELLS_PER_M, (y_max - gty.y())*OccupancyTile::CELLS_PER_M);
                cvLine(composite_img, g, gx, cvScalar(0,0,0));
                cvLine(composite_img, g, gy, cvScalar(0,0,0));
            }

            cvShowImage("MapBuilder", composite_img);

            cvReleaseImage(&map_img);
            cvReleaseImage(&composite_img);

            return true;
        } //end Build

        bool MapBuilder::WritePersistent()
        {
            //dump mapping data to disk
            //simple ASCII format, one line per object (so we can edit/cut/paste data)
            //submap image files have already been written to data/submaps
            char data_filepath[255];
            sprintf(data_filepath, "%s/map_state/%s.txt", DATA_PATH, generated_map_uuid.c_str());

            ofstream outfile(data_filepath, ios_base::trunc);

            if (!outfile)
            {
                SIMPLELOGGER_ERROR("WritePersistent: cant open %s for writing", data_filepath);
                return false;
            }

            SIMPLELOGGER_DEBUG("WritePersistent: writing data to %s.", data_filepath);

            outfile << PERSIST_FILE_HEADER << endl;

            BOOST_FOREACH(Submap* submap, map->GetSubmapList())
            {
                //submap:
                submap->WriteToAsciiStream(outfile);

                //submap pose
                if (submap->GetPose()) submap->GetPose()->WriteToAsciiStream(outfile);

                //submap ground truth
                if (submap->GetGroundTruth()) submap->GetGroundTruth()->WriteToAsciiStream(outfile);

                //base constraints:
                BOOST_FOREACH(SubmapConstraint* constraint, submap->GetBaseConstraints())
                constraint->WriteToAsciiStream(outfile);
            }

            SIMPLELOGGER_DEBUG("WritePersistent: done.");
            return true;
        }//end persist

        bool MapBuilder::ReadPersistent(char * data_filepath)
        {
#if 0
            //load mapping data from disk
            ifstream infile(data_filepath);
            if (!infile)
            {
                SIMPLELOGGER_ERROR("WritePersistent: cant open %s for reading.", data_filepath);
                return false;
            }

            string line;
            int countSubmaps = 0, countSubmapPoses = 0, countConstraints = 0, countGroundTruths = 0;

            if (!getline(infile, line) || line != PERSIST_FILE_HEADER)
            {
                SIMPLELOGGER_ERROR("WritePersistent: incorrect header line or no data.");
                return false;
            }

            //loop over lines: first load submaps:
            while (getline(infile, line))
            {
                stringstream lineStream(line);
                string token;
                lineStream >> token;

                if (token == "Submap")
                {
                    Submap *submap = new Submap(lineStream);
                    map->SetSubmap(submap);
                    countSubmaps++;
                }
            }

            //reset and reload everything else
            infile.clear();
            infile.seekg(0);
            getline(infile, line);

            while (getline(infile, line))
            {
                stringstream lineStream(line);
                string token;
                lineStream >> token;

                if (token == "SubmapPose")
                {
                    SubmapPose *pose = new SubmapPose(lineStream);
                    map->SetSubmapPose(pose);
                    countSubmapPoses++;
                }

                if (token == "SubmapGroundTruth")
                {
                    SubmapGroundTruth *gt = new SubmapGroundTruth(lineStream);
                    map->SetSubmapGroundTruth(gt);
                    countGroundTruths++;
                }

                if (token == "SubmapConstraint")
                {
                    SubmapConstraint *constraint = new SubmapConstraint(lineStream);
                    map->SetSubmapConstraint(constraint);
                    countConstraints++;
                }
            }

            SIMPLELOGGER_INFO("ReadPersistent added %d submaps, %d poses, %d constraints, %d ground truthes to map.",
                    countSubmaps, countSubmapPoses, countConstraints, countGroundTruths);
#endif
            return true;
        }

        bool MapBuilder::MatchSubmaps()
        {
            //same code for wambots and GCS
            //if GCS, or no current_submap randomly choose a submap to match

            if (!matching_framebuffer)
                SIMPLELOGGER_INFO("Matching: No matching framebuffer, skipping.");
            if (!map->submaps_list.size())
                SIMPLELOGGER_INFO("Matching: No submaps, skipping.");

            /*
                std::list<Submap*> base_submaps;

                if (is_WAMBOT && current_submap && current_submap->GetPose()) {
                    if (current_submap->GetCellCountOccupied()<submapMatchingMinimumOccupiedCellCount ) {
                        SIMPLELOGGER_INFO( "Matching: Not enough occupied cells to match current submap" );
                                }
                    else
                        base_submaps.push_back( current_submap );
                }

                randomly chose up to 10 more base submaps to attempt matching from
                int pick = map->GetSubmapList().size();
                if (pick>10) pick=10;


                for (int i=0; i<pick; i++)
                {
                    //random index:
                    int index = rand() % map->GetSubmapList().size();
                    //get the submap
                    list<Submap*>::iterator iter = map->GetSubmapList().begin();
                    std::advance(iter,index);
                    Submap* potential_base_submap = *iter;
                    //checks: is it already picked?
                    list<Submap*>::iterator result = find(base_submaps.begin(), base_submaps.end(), potential_base_submap );
                    if (result != base_submaps.end()) continue;
                    //checks: must have occupied cells:
                    if (potential_base_submap->GetCellCountOccupied()<submapMatchingMinimumOccupiedCellCount ) continue;
                    //checks: must have a pose:
                    if (!potential_base_submap->GetPose()) continue;
                    //checks: must be linked to other submaps:
                    if (potential_base_submap->dont_match) continue;
                    if (!potential_base_submap->IsConstrained()) continue;
                    //we've got a base submap:
                    base_submaps.push_back( potential_base_submap );
                }

                if (base_submaps.size()<1) {
                    SIMPLELOGGER_INFO( "Matching: No base submaps found to match" );
                    return false;
                }

                std::list<SubmapPair> base_target_submap_pairs;

                loop over each base submap, search for target submaps to test against:
                BOOST_FOREACH(Submap* base_submap, base_submaps)
                {
                    find a target submap to match with
                    CPose2D  base_pose = base_submap->GetPose()->GetMean();

                    loop over each potential target submap:
                    BOOST_FOREACH(Submap* target_submap, map->GetSubmapList() )
                    {
                        if (base_submap == target_submap) continue;

                        //must have pose, else keep searching
                        if (!target_submap->GetPose()) continue;
                        CPose2D target_pose = target_submap->GetPose()->GetMean();

                        //must be linked to other submaps:
                        if (!target_submap->IsConstrained()) continue;

                        if (target_submap->dont_match) continue;

                        //must be within Xm of eachother, else keep searching
                        if ((base_pose).sqrDistanceTo(target_pose) > square(10.0) ) continue;

                        //CPoint2D test_size = target_submap->GetSize();
                        //target submap must be larger than 10sqm
                        //if (test_size.x()*test_size.y()<10.0) continue;
                        if (target_submap->GetCellCountOccupied()<submapMatchingMinimumOccupiedCellCount ) continue;

                        //now, target MAY overlap current.
                        //todo test for overlap of extents

                        //todo check for existing constraints
                        SubmapConstraint* existing_constraint = base_submap->GetSubmapConstraint( target_submap );

                        //if the current constraint is less confident than existing matching+odometry, skip
                        if (existing_constraint && (existing_constraint->confidence>0.7)) continue;

                        //check if we've tested this pair, contatenate UUID's for key
                        bool tested = submap_submap_match_tested[ base_submap->GetUUID()+target_submap->GetUUID() ];
                        if (tested) continue;

                        //store the pair
                        base_target_submap_pairs.push_back( SubmapPair(base_submap,target_submap) );

                        //stop looking if we've found 5
                        if (base_target_submap_pairs.size()>5) break;
                    }
                }

                //end for base submaps
                if (base_target_submap_pairs.size()<1)
                {
                    SIMPLELOGGER_INFO( "Matching: No base-target submap pairs found." );
                    return false;
                }
            */

/*
            ofstream scoresFile;
            scoresFile.open("scores", ios::app);

            ofstream iterationFile;
            iterationFile.open("iteration", ios::app);
*/

            CTicTac total_time;
            total_time.Tic();
            CTicTac setup_time;
            setup_time.Tic();

            //bind framebuffer, setup ortho projection, clear buffer and load GLSL shader
            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, matching_framebuffer);
            CheckOpenGlError("glBindFramebufferEXT matching_framebuffer");

            //glPushAttrib(GL_VIEWPORT_BIT);

            glViewport(0, 0, matching_pixel_width, matching_pixel_height);
            glDisable(GL_DEPTH_TEST);
            //glClear(GL_COLOR_BUFFER_BIT ); //GL_DEPTH_BUFFER_BIT); not needed.

            glUseProgram(submap_matcher_sp);
            CheckOpenGlError("glUseProgram Submap matching program");

            stats["match setup time"] = setup_time.Tac();
            stats["match submap total count"] = map->submaps_list.size();

            if (matching_submap_iterator == map->submaps_list.rend()) //if at end from last search, restart
                matching_submap_iterator = map->submaps_list.rbegin();

            //loop performing submap matches
            TTimeStamp ts_end = now()+(TTimeStamp)( SUBMAP_MATCHING_TIME_INCREMENT_MAX * TS_SEC ); //0.1 sec

            //SUBMAP_MATCHING_TIME_MAX
            while ((now()<ts_end) && (matching_submap_iterator != map->submaps_list.rend()))
            {
                Submap *base_submap = *matching_submap_iterator;
                CPose2D base_pose = base_submap->GetPose()->GetMean();

                if (!base_submap->IsConstrained()) { matching_submap_iterator++; continue; }

                //TODO spatial index:
                list<Submap*>::iterator it = map->submaps_list.begin();

                while (it!=map->submaps_list.end() && (now()<ts_end))
                {
                    //BOOST_FOREACH(Submap* target_submap, ) {
                    Submap* target_submap = *it++;
                    if (base_submap == target_submap) continue;

                    //must have pose, else keep searching
                    if (!target_submap->GetPose()) continue;
                    CPose2D target_pose = target_submap->GetPose()->GetMean();
                    CPose2D rel_pose = target_pose - base_pose;

                    //must be linked to other submaps:
                    if (!target_submap->IsConstrained()) continue;

                    if (target_submap->dont_match) continue;

                    //must be within Xm of eachother, else keep searching
                    if (rel_pose.norm() > MATCH_REL_POSE_THRESH) continue;

                    //CPoint2D test_size = target_submap->GetSize();
                    //target submap must be larger than 10sqm
                    //if (test_size.x()*test_size.y()<10.0) continue;
                    if (target_submap->GetCellCountOccupied() < submapMatchingMinimumOccupiedCellCount) continue;

                    //now, target MAY overlap current
                    //todo test for overlap of extents

                    //cout << base_submap->GetUUID()  << target_submap->GetUUID()  << endl;

                    //todo check for existing constraints
                    SubmapConstraint* existing_constraint = base_submap->GetSubmapConstraint(target_submap);

                    //if the current constraint is less confident than existing matching+odometry, skip
                    if (existing_constraint && (existing_constraint->confidence > 0.7)) continue;

                    //check if we've tested this pair, contatenate UUID's for key
                    SubmapPair this_pair = SubmapPair(base_submap, target_submap);
                    CPose2D tested_rel_pose = submap_submap_match_tested[ this_pair ];

                    //compare the relative pose when we tested, and currently
                    //if the delta has changed by more than the search size, retest
                    CPose2D delta = tested_rel_pose - rel_pose;
                    if ((delta.norm() < MATCH_SEARCH_WIDTH / 2) && (fabs(delta.phi()) < MATCH_SEARCH_ANGLE / 2)) continue;

                    //now we have base_submap and target_submap, attempt brute-force match
                    GLuint *base_texture = occupancy_texture[ base_submap->GetUUID()];
                    GLuint *target_texture = occupancy_texture[target_submap->GetUUID()];
                    //these tex aren't used

                    if (!base_texture || !target_texture)
                    {
                            //SIMPLELOGGER_DEBUG( "Matching: submap has no texture, skipping match." );
                            continue;
                    }

                    ///////set this pair as tested///////
                    //will not be tested again unless their relative poses change
                    submap_submap_match_tested[ this_pair ] = rel_pose;

                    //Now begin the actual testing
                    SIMPLELOGGER_DEBUG("Matching: submap base:%s target:%s.", base_submap->GetUUID().substr(0, 6).c_str(), target_submap->GetUUID().substr(0, 6).c_str());

                    //big recursive search to get existing constraint covariance:
                    existing_constraint = base_submap->FindRecursiveSubmapConstraint(target_submap);
                    //this is copied, delete at end

                    //test hack todo
                    //trim tex border
                    setup_time.Tic();
                    const int SCALE = 2;
                    IplImage* base_occupancy_img = base_submap->GetOccupancyData();
                    IplImage* base_temp_img_full = cvCreateImage(cvSize(base_occupancy_img->width, base_occupancy_img->height), IPL_DEPTH_8U, 1);
                    IplImage* base_temp_img = cvCreateImage(cvSize(base_occupancy_img->width / SCALE, base_occupancy_img->height / SCALE), IPL_DEPTH_8U, 1);

                    cvSetImageCOI(base_occupancy_img, 2); //green channel
                    cvCopy(base_occupancy_img, base_temp_img_full);
                    cvResize(base_temp_img_full, base_temp_img);
                    cvSetImageCOI(base_occupancy_img, 0);

                    base_texture = new GLuint;
                    glGenTextures(1, base_texture);
                    glBindTexture(GL_TEXTURE_2D, *base_texture);
                    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                    //glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, OccupancyTile::WIDTH, OccupancyTile::HEIGHT,0, GL_LUMINANCE, GL_UNSIGNED_BYTE, tile->GetData() );
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, base_temp_img->width, base_temp_img->height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, base_temp_img->imageData);
                    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, occupancy_img->width, occupancy_img->height,0, GL_RGB, GL_UNSIGNED_BYTE, occupancy_img->imageData );
                    //glTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB, occupancy_img->width, occupancy_img->height,0, GL_RGB, GL_UNSIGNED_BYTE, occupancy_img->imageData );
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); //GL_LINEAR
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); //GL_NEAREST
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

                    IplImage* target_occupancy_img = target_submap->GetOccupancyData();
                    IplImage* target_temp_img_full = cvCreateImage(cvSize(target_occupancy_img->width, target_occupancy_img->height), IPL_DEPTH_8U, 1);
                    IplImage* target_temp_img = cvCreateImage(cvSize(target_occupancy_img->width / SCALE, target_occupancy_img->height / SCALE), IPL_DEPTH_8U, 1);

                    cvSetImageCOI(target_occupancy_img, 2); //green channel
                    cvCopy(target_occupancy_img, target_temp_img_full);
                    cvResize(target_temp_img_full, target_temp_img);
                    cvSetImageCOI(target_occupancy_img, 0);

                    target_texture = new GLuint;
                    glGenTextures(1, target_texture);
                    glBindTexture(GL_TEXTURE_2D, *target_texture);
                    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                    //glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, OccupancyTile::WIDTH, OccupancyTile::HEIGHT,0, GL_LUMINANCE, GL_UNSIGNED_BYTE, tile->GetData() );
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, target_temp_img->width, target_temp_img->height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, target_temp_img->imageData);
                    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, occupancy_img->width, occupancy_img->height,0, GL_RGB, GL_UNSIGNED_BYTE, occupancy_img->imageData );
                    //glTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB, occupancy_img->width, occupancy_img->height,0, GL_RGB, GL_UNSIGNED_BYTE, occupancy_img->imageData );
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); //GL_LINEAR
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); //GL_NEAREST
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

                    stats["match setup time"] += setup_time.Tac();
                    CTicTac correlation_time;
                    correlation_time.Tic();

                    glActiveTexture(GL_TEXTURE0);
                    glEnable(GL_TEXTURE_2D);
                    glBindTexture(GL_TEXTURE_2D, *base_texture);
                    glUniform1i(submap_matcher_base_submap_tex, 0);

                    glActiveTexture(GL_TEXTURE1);
                    glEnable(GL_TEXTURE_2D);
                    glBindTexture(GL_TEXTURE_2D, *target_texture);
                    glUniform1i(submap_matcher_target_submap_tex, 1);

                    //set submap's extents as a uniform
                    glUniform2f(submap_matcher_base_bottom_left, base_submap->GetExtentsTopLeft().x(), base_submap->GetExtentsBottomRight().y());
                    glUniform2f(submap_matcher_base_top_right, base_submap->GetExtentsBottomRight().x(), base_submap->GetExtentsTopLeft().y());
                    glUniform2f(submap_matcher_target_bottom_left, target_submap->GetExtentsTopLeft().x(), target_submap->GetExtentsBottomRight().y());
                    glUniform2f(submap_matcher_target_top_right, target_submap->GetExtentsBottomRight().x(), target_submap->GetExtentsTopLeft().y());

                    //set initial offset guess
                    CPose2D initial_offset = target_pose - base_pose;
                    SIMPLELOGGER_DEBUG("Matching: initial offset: %1.1f, %1.1f, %1.1f deg.", initial_offset.x(), initial_offset.y(), R2D(initial_offset.phi()));

                    SIMPLELOGGER_DEBUG("Matching: base   occ:%d free:%d unk:%d", base_submap->GetCellCountOccupied(), base_submap->GetCellCountFree(), base_submap->GetCellCountUnknown());
                    SIMPLELOGGER_DEBUG("Matching: target occ:%d free:%d unk:%d", target_submap->GetCellCountOccupied(), target_submap->GetCellCountFree(), target_submap->GetCellCountUnknown());

                    CPose2D best_offset;
                    double score = 1e99;
                    double score_threshold;
                    Matrix3d covariance;
                    double avg_count_occupied = (base_submap->GetCellCountOccupied() + target_submap->GetCellCountOccupied()) / 2;
                    double search_width = MATCH_SEARCH_WIDTH;
                    double search_angle = MATCH_SEARCH_ANGLE;

                    if (existing_constraint)
                    {
                        double mult = 1.0 / (pow(2 * M_PI, 1.5) * sqrt(existing_constraint->constraint_covariance.determinant()));
                        score_threshold = submapMatchingScoreThresholdWithPrior / 100.0 * avg_count_occupied * mult;

                        search_width = MAX(sqrt(existing_constraint->constraint_covariance(0, 0))*SINGLE_MATCH_COUNT,
                                sqrt(existing_constraint->constraint_covariance(1, 1))*SINGLE_MATCH_COUNT);
                        search_angle = sqrt(existing_constraint->constraint_covariance(2, 2))*SINGLE_MATCH_COUNT;
                    }
                    else
                    {
                        score_threshold = submapMatchingScoreThreshold / 100.0 * avg_count_occupied;
                    }

                    score = DoSingleMatch(search_width, search_width, search_angle, &initial_offset, &best_offset, &covariance, base_submap, target_submap, existing_constraint);
                    SIMPLELOGGER_DEBUG("Matching: best_offset: %1.1f, %1.1f, %1.1f deg. score: %1.1f", best_offset.x(), best_offset.y(), R2D(best_offset.phi()), score);

                    if (score > score_threshold && search_width > 2.5)
                    {
                        //todo more precise search heuristic: follow peaks
                        //wide search match passed threshold, perform narrow search: (4)
                        score = DoSingleMatch(search_width / 4, search_width / 4, search_angle / 4, &best_offset, &best_offset, &covariance, base_submap, target_submap, existing_constraint);
                        SIMPLELOGGER_DEBUG("Matching: refined best_offset: %1.1f, %1.1f, %1.1f deg. score: %1.1f", best_offset.x(), best_offset.y(), R2D(best_offset.phi()), score);
                    }

/*
                    int iteration = 0;

                    score = DoSingleMatch(search_width, search_width, search_angle, &initial_offset, &best_offset, &covariance, base_submap, target_submap, existing_constraint);

                    if (search_width > 2.5)
                    {
                        for (int i = 2; i < 7; i++)
                        {
                            iteration++;

                            CPose2D tempOffset;
                            Matrix3d tempCovariance;

                            double currentScore = DoSingleMatch(search_width / (i), search_width / (i), search_angle / (i), &best_offset, &tempOffset, &tempCovariance, base_submap, target_submap, existing_constraint);
                            double percent = ((currentScore - score) / currentScore);

                            if (percent > 0.15)
                            {
                                covariance = tempCovariance;
                                best_offset = tempOffset;

                                score = currentScore;
                            }
                            else
                            {
                                break;
                            }
                        }
                    }
*/
                    //scoresFile << score << "\n";
                    //iterationFile << iteration << "\n";

                    SIMPLELOGGER_ERROR("Matching: final best_offset: %1.5f, %1.5f, %1.5f deg. score: %1.5f", best_offset.x(), best_offset.y(), R2D(best_offset.phi()), score);

                    //Record the scores for the most recent submap
                    if (base_submap == most_recent_submap)
                    {
                        submap_scores.push_back(score);
                        submap_scores.push_back(score_threshold);
                    }

                    if (score > score_threshold)
                    {
                        //consider this a good match:
                        SIMPLELOGGER_DEBUG("Matching: match found score %1.1f > %1.1f, testing constraint error <<<<<<<<<<<<< ", score, score_threshold); //, sending SubmapConstraint." );

                        SubmapConstraint *constraint = new SubmapConstraint(CreateUUID(),
                                base_submap->GetUUID(), target_submap->GetUUID(), best_offset,
                                covariance, CONSTRAINT_TYPE_SUBMAP_MATCHER, existing_constraint ? 0.90 : 0.60);

                        constraint->base_submap = base_submap;
                        constraint->target_submap = target_submap;

                        bool accept = OptimiseSubmaps(constraint);

                        if (accept)
                        {
                            map->SetSubmapConstraint(constraint);
                            SIMPLELOGGER_INFO("Matching: optimiser accepted new constraint <<<<<<<<<<<<<");
                        }
                        else
                        {
                            delete constraint;
                            SIMPLELOGGER_INFO("Matching: optimiser ignored bad constraint <<<<<<<<<<<<<");
                        }
                    }
                    else
                    {
                        SIMPLELOGGER_DEBUG("Matching: match found score %1.1f < %1.1f, ignoring bad match. <<<<<<<<<<<<<", score, score_threshold); //, sending SubmapConstraint." );
                    }
                    stats["match correlation time"] += correlation_time.Tac();

                    setup_time.Tic();
                    if (existing_constraint) delete existing_constraint;

                    //todo unhack
                    glDeleteTextures(1, base_texture);
                    CheckOpenGlError("glDeleteTextures( 1, base_texture )");
                    glDeleteTextures(1, target_texture);
                    CheckOpenGlError("glDeleteTextures( 1, target_texture )");
                    delete base_texture;
                    delete target_texture;
                    cvReleaseImage(&base_temp_img);
                    cvReleaseImage(&base_temp_img_full);
                    cvReleaseImage(&target_temp_img);
                    cvReleaseImage(&target_temp_img_full);
                    stats["match setup time"] += setup_time.Tac();

                }//end foreach potential target submap

                matching_submap_iterator++; //onto next base

            }//end while scanning in reverse over base submaps

            stats["match total time"] += total_time.Tac();
            stats["match search time"] = stats["match total time"] - stats["match setup time"] - stats["match correlation time"];

            SIMPLELOGGER_DEBUG("Matching done.");

            //scoresFile.close();
            //iterationFile.close();

            //return true if we finished the search/match
            return matching_submap_iterator == map->submaps_list.rend();

        }//end matchsubmaps

        double MapBuilder::DoSingleMatch(float x_range, float y_range, float phi_range, CPose2D *initial_offset, CPose2D *measured_offset,
                Matrix3d *measured_cov, Submap* base_submap, Submap* target_submap, SubmapConstraint * existing_constraint)
        {
            CTicTac tictac; //timer
            tictac.Tic();
            //perform single brute-force matching
            SIMPLELOGGER_DEBUG("Matching: x_range:%1.1f y_range:%1.1f phi_range:%1.1f deg", x_range, y_range, R2D(phi_range));

            glUniform3f(submap_matcher_initial_offset, initial_offset->x(), initial_offset->y(), initial_offset->phi());

            //range of pose offsets to test
            float x_min = initial_offset->x() - x_range / 2.0;
            float x_max = initial_offset->x() + x_range / 2.0;
            float y_min = initial_offset->y() - y_range / 2.0;
            float y_max = initial_offset->y() + y_range / 2.0;
            float phi_min = initial_offset->phi() - phi_range / 2.0; //2.0 was the original
            //float phi_max = initial_offset->phi() + phi_range/2.0;
            float phi_step = phi_range / ((SINGLE_MATCH_COUNT * SINGLE_MATCH_COUNT) - 1);
            // 3x3 = 9 tests
            int count = matching_pixel_width * matching_pixel_height;

            CTicTac gpu_time;
            gpu_time.Tic();

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0, x_range * SINGLE_MATCH_COUNT,
                    0, y_range * SINGLE_MATCH_COUNT,
                    -1.0f, 1.0f);
            glMatrixMode(GL_MODELVIEW);

            //SIMPLELOGGER_DEBUG( "Matching: GLSL start" );
            for (int i = 0; i < (SINGLE_MATCH_COUNT * SINGLE_MATCH_COUNT); i++)
            {
                float phi = phi_min + (i * phi_step);
                glLoadIdentity();
                glTranslatef(-x_min + (i % SINGLE_MATCH_COUNT) * x_range, -y_min + (i / SINGLE_MATCH_COUNT) * y_range, -phi);
                //quad defines per-pixel test values for submap correlation:
                glBegin(GL_QUADS);
                glVertex3f(x_min, y_max, phi); //Top left
                glVertex3f(x_max, y_max, phi); //Top right
                glVertex3f(x_max, y_min, phi); //Bottom right
                glVertex3f(x_min, y_min, phi); //Bottom left
                glEnd();
            }
            //sleep(0);
            glFinish();

            stats["match correlation gpu time"] += gpu_time.Tac();
            stats["match correlation gpu count"] += count;

            //SIMPLELOGGER_DEBUG( "Matching: GLSL finished" );
            IplImage* matched_img = cvCreateImage(cvSize(matching_pixel_width, matching_pixel_height), IPL_DEPTH_32F, 1);
            IplImage* matched_img_output = cvCreateImage(cvSize(matching_pixel_width, matching_pixel_height), IPL_DEPTH_8U, 1);
            //SIMPLELOGGER_DEBUG( "Matching: reading array" );
            glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
            glReadPixels(0, 0, matching_pixel_width, matching_pixel_height, GL_ALPHA, GL_FLOAT, matched_img->imageData);
            glFinish();
            //SIMPLELOGGER_DEBUG( "Matching: read done" );

            const int w = matching_pixel_width / SINGLE_MATCH_COUNT;
            const int h = matching_pixel_height / SINGLE_MATCH_COUNT;
            double minVal, match_score;
            CvPoint minLoc, maxLoc;
            cvMinMaxLoc(matched_img, &minVal, &match_score, &minLoc, &maxLoc);
            //map image pixel back to solution space:
            float x_mean = x_min + x_range * (maxLoc.x % w) / (float) w;
            float y_mean = y_min + y_range * (maxLoc.y % h) / (float) h; //y coord is flipped twice already
            //todo verify pixel center
            int j = maxLoc.x / w + SINGLE_MATCH_COUNT * (maxLoc.y / h);
            float phi_mean = phi_min + j*phi_step;
            *measured_offset = CPose2D(x_mean, y_mean, phi_mean);

#if WRITE_MATCHING
            //write matching image to disk
            string filepath = format("%s/%s---%s---matching-%1.1fx%1.1fm---%1.1f.png", matches_path, base_submap->GetUUID().c_str(), target_submap->GetUUID().c_str(), x_range, y_range, match_score);
            cvConvertScaleAbs(matched_img, matched_img_output, 255.0 / match_score);
            //todo plot cov on image
            //http://www.nr.com/CS395T/lectures2008/9-MultivariateNormal.pdf
            cvCircle(matched_img_output, maxLoc, 8, cvScalar(255));
            cvConvertImage(matched_img_output, matched_img_output, CV_CVTIMG_FLIP);
            cvSaveImage(filepath.c_str(), matched_img_output);
            SIMPLELOGGER_DEBUG("Saved matching (likelihood) data to %s", filepath.c_str());
#endif

            //test todo
            if (existing_constraint)
            { //we have a prior

                IplImage* prior_img = cvCreateImage(cvSize(matching_pixel_width, matching_pixel_height), IPL_DEPTH_32F, 1);
                IplImage* prior_img_output = cvCreateImage(cvSize(matching_pixel_width, matching_pixel_height), IPL_DEPTH_8U, 1);
                Vector3d mean(existing_constraint->constraint_mean.x(), existing_constraint->constraint_mean.y(), existing_constraint->constraint_mean.phi());
                Matrix3d prec = existing_constraint->constraint_precision;
                double mult = 1.0 / (pow(2 * M_PI, 1.5) * sqrt(existing_constraint->constraint_covariance.determinant()));

                for (int yy = 0; yy < matching_pixel_height; yy++)
                {
                    float *prior_buffer = (float*) &prior_img->imageData[yy * prior_img->widthStep]; //note opencv image data is char*                    
                    float *likelihood_buffer = (float*) &matched_img->imageData[yy * matched_img->widthStep]; //cast to float pointers

                    for (int xx = 0; xx < matching_pixel_width; xx++)
                    {
                        //todo optimise
                        float x = x_min + x_range * (xx % w) / (float) w;
                        float y = y_min + y_range * (yy % h) / (float) h; //y coord is flipped twice already
                        int j = xx / w + SINGLE_MATCH_COUNT * (yy / h);
                        float phi = phi_min + j*phi_step;
                        Vector3d x_i_minus_mean = Vector3d(x, y, phi) - mean;
                        double prior = mult * exp(-0.5 * (double) (x_i_minus_mean.transpose() * prec * x_i_minus_mean));
                        prior_buffer[ xx ] = prior;
                        likelihood_buffer[ xx ] *= prior;
                    }
                }
#if WRITE_MATCHING
                string filepath = format("%s/%s---%s---prior-%1.1fx%1.1fm---%1.1f.png", matches_path, base_submap->GetUUID().c_str(), target_submap->GetUUID().c_str(), x_range, y_range, mult);
                cvConvertScaleAbs(prior_img, prior_img_output, 255.0 / mult);
                cvConvertImage(prior_img_output, prior_img_output, CV_CVTIMG_FLIP);
                cvSaveImage(filepath.c_str(), prior_img_output);
#endif

                cvReleaseImage(&prior_img);
                cvReleaseImage(&prior_img_output);

                //the matched image has now been multiplied by the odometry prior
                //find the new maximum
                cvMinMaxLoc(matched_img, &minVal, &match_score, &minLoc, &maxLoc);
                //map image pixel back to solution space:
                x_mean = x_min + x_range * (maxLoc.x % w) / (float) w;
                y_mean = y_min + y_range * (maxLoc.y % h) / (float) h; //y coord is flipped twice already
                //todo verify pixel center
                int j = maxLoc.x / w + SINGLE_MATCH_COUNT * (maxLoc.y / h);
                float phi_mean = phi_min + j*phi_step;
                //measured offset is now the posterior
                *measured_offset = CPose2D(x_mean, y_mean, phi_mean);
                //write posterior image to disk

#ifdef WRITE_MATCHING
                filepath = format("%s/%s---%s---posterior-%1.1fx%1.1fm---%1.1f.png", matches_path, base_submap->GetUUID().c_str(), target_submap->GetUUID().c_str(), x_range, y_range, match_score);
                cvConvertScaleAbs(matched_img, matched_img_output, 255.0 / match_score);
                //todo plot cov on image
                //http://www.nr.com/CS395T/lectures2008/9-MultivariateNormal.pdf
                cvCircle(matched_img_output, maxLoc, 8, cvScalar(255));
                cvConvertImage(matched_img_output, matched_img_output, CV_CVTIMG_FLIP);
                cvSaveImage(filepath.c_str(), matched_img_output);
                //SIMPLELOGGER_DEBUG( "Saved likelihood data to %s", filepath.c_str() );
#endif
            }


            //		if (measured_cov) {
            //compute covariance http://en.wikipedia.org/wiki/Sample_covariance_matrix#Weighted_samples
            //also: april.eecs.umich.edu/papers/details.php?name=olson2009icra
            //MatrixXf K(3,3);
            Matrix3d K = Matrix3d::Zero();
            //K.Zero();
            //K << 0,0,0,0,0,0,0,0,0;
            //Vector3f u;
            //u << 0,0,0;
            double s = 0.0;
            double t = 0.0;

            for (int yy = 0; yy < matching_pixel_height; yy++)
            {
                float *matched_buffer = (float*) &matched_img->imageData[yy * matched_img->widthStep]; //cast to float pointers

                for (int xx = 0; xx < matching_pixel_width; xx++)
                {
                    //todo optimise
                    float x = x_min + x_range * (xx % w) / (float) w;
                    float y = y_min + y_range * (yy % h) / (float) h; //y coord is flipped twice already
                    int j = xx / w + SINGLE_MATCH_COUNT * (yy / h);
                    float phi = phi_min + j*phi_step;
                    double P_x_i = matched_buffer[ xx ];
                    Vector3d x_i(x - x_mean, y - y_mean, phi - phi_mean);
                    K += x_i * x_i.transpose() * P_x_i;
                    //u += x_i * P_x_i;
                    t += P_x_i;
                    s += P_x_i*P_x_i;
                }
            }
            //Matrix3f cov = 1.0/s*K - 1.0/(s*s)*u*u.transpose();
            *measured_cov = K / (t - s / t);
            //todo verify
            //}

            SelfAdjointEigenSolver<Matrix2d> eigensolver(measured_cov->block < 2, 2 > (0, 0));
            Vector2d e = eigensolver.eigenvalues();
            double ratio = e[0] > e[1] ? e[1] / e[0] : e[0] / e[1];

            SIMPLELOGGER_DEBUG("Matching: max val: %1.1f, cov eig ratio: %1.2f", match_score, ratio);
            match_score *= ratio;
            SIMPLELOGGER_DEBUG("Matching: best: x:%1.1f y:%1.1f phi:%1.1f deg. Score: %1.1f", x_mean, y_mean, R2D(phi_mean), match_score);
            //		if (measured_cov) {
            //			SIMPLELOGGER_DEBUG("  [ %1.3f %1.3f %1.3f,", (*measured_cov)(0,0),(*measured_cov)(0,1),(*measured_cov)(0,2) );
            //			SIMPLELOGGER_DEBUG("    %1.3f %1.3f %1.3f,", (*measured_cov)(1,0),(*measured_cov)(1,1),(*measured_cov)(1,2) );
            //			SIMPLELOGGER_DEBUG("    %1.3f %1.3f %1.3f ]",(*measured_cov)(2,0),(*measured_cov)(2,1),(*measured_cov)(2,2) );
            //		}
            SIMPLELOGGER_DEBUG("Matching: tested %d matches, %1.0f tests per second.", count, (float) count / tictac.Tac());
            cvReleaseImage(&matched_img);
            cvReleaseImage(&matched_img_output);

            return match_score;
        }

                /*
                bool MapBuilder::CheckAddGroundTruthConstraint( org::wambot::datamodel::HmiMapConstraint *hmiMapConstraint ) {
                        SIMPLELOGGER_INFO("HmiMapConstraint on map:%s current map:%s", hmiMapConstraint->generated_map_uuid, generated_map_uuid.c_str() );

                        double northing, easting;
                        LatLonToUtmWGS84( hmiMapConstraint->generated_map_first_coordinate.latitude, hmiMapConstraint->generated_map_first_coordinate.longitude, northing, easting, utm_zone);
                        CPoint2D generated_map_first_point(easting,northing);
                        LatLonToUtmWGS84( hmiMapConstraint->generated_map_second_coordinate.latitude, hmiMapConstraint->generated_map_second_coordinate.longitude, northing, easting, utm_zone);
                        CPoint2D generated_map_second_point(easting,northing);
                        CPoint2D generated_map_delta = generated_map_second_point-generated_map_first_point;
                        double generated_map_phi = atan2( generated_map_delta.y(),generated_map_delta.x() );

                        LatLonToUtmWGS84( hmiMapConstraint->ground_truth_map_first_coordinate.latitude, hmiMapConstraint->ground_truth_map_first_coordinate.longitude, northing, easting, utm_zone);
                        CPoint2D ground_truth_map_first_point(easting,northing);
                        LatLonToUtmWGS84( hmiMapConstraint->ground_truth_map_second_coordinate.latitude, hmiMapConstraint->ground_truth_map_second_coordinate.longitude, northing, easting, utm_zone);
                        CPoint2D ground_truth_map_second_point(easting,northing);
                        CPoint2D ground_truth_map_delta = ground_truth_map_second_point-ground_truth_map_first_point;
                        double ground_truth_map_phi = atan2( ground_truth_map_delta.y(),ground_truth_map_delta.x() );

                        SIMPLELOGGER_INFO("     Submap: %s %1.2f,%1.2f,%1.2f deg. -> %1.2f,%1.2f,%1.2f deg.", hmiMapConstraint->submap_uuid,
                                        generated_map_first_point.x(), generated_map_first_point.y(),      R2D(generated_map_phi),
                                        ground_truth_map_first_point.x(), ground_truth_map_first_point.y(),R2D(ground_truth_map_phi) );

                        if (string(hmiMapConstraint->generated_map_uuid) != generated_map_uuid) {
                                SIMPLELOGGER_WARN("HmiMapConstraint has different generated_map_uuid, ignoring.");
                                return false;
                        }

                        Submap *submap = map->GetSubmap(hmiMapConstraint->submap_uuid);
                        if (!submap) {
                                SIMPLELOGGER_WARN("HmiMapConstraint has invalid submap_uuid, ignoring.");
                                return false;
                        }

                        //got a submap, send constraint
                        CPose2D offset = generated_map_first_point - submap->GetPose()->GetMean();
                        CPose2D delta( ground_truth_map_first_point - generated_map_first_point );
                        delta.phi( ground_truth_map_phi - generated_map_phi   );
                        CPose2D ground_truth = submap->GetPose()->GetMean() + offset + delta + offset;
                        submapGroundTruthMessage->submap_uuid = (char*)submap->GetUUID().c_str();
                        submapGroundTruthMessage->mean[0] = ground_truth.x();
                        submapGroundTruthMessage->mean[1] = ground_truth.y();
                        submapGroundTruthMessage->mean[2] = ground_truth.phi();
                        submapGroundTruthMessage->covariance[0] = sqr(HMI_GROUND_TRUTH_CONSTRAINT_ERROR_LIN_STDEV);
                        submapGroundTruthMessage->covariance[4] = sqr(HMI_GROUND_TRUTH_CONSTRAINT_ERROR_LIN_STDEV);
                        submapGroundTruthMessage->covariance[8] = sqr(HMI_GROUND_TRUTH_CONSTRAINT_ERROR_ANG_STDEV);
                        submapGroundTruthTopicWrapper->write(submapGroundTruthMessage);
                        SIMPLELOGGER_DEBUG("Sent submap ground truth, mean: %1.2f, %1.2f, %1.2f deg.", submapGroundTruthMessage->mean[0], submapGroundTruthMessage->mean[1], R2D(submapGroundTruthMessage->mean[2])  );
                        SIMPLELOGGER_DEBUG("  [ %1.3f %1.3f %1.3f,", submapGroundTruthMessage->covariance[0],submapGroundTruthMessage->covariance[1],submapGroundTruthMessage->covariance[2] );
                        SIMPLELOGGER_DEBUG("    %1.3f %1.3f %1.3f,", submapGroundTruthMessage->covariance[3],submapGroundTruthMessage->covariance[4],submapGroundTruthMessage->covariance[5] );
                        SIMPLELOGGER_DEBUG("    %1.3f %1.3f %1.3f ]",submapGroundTruthMessage->covariance[6],submapGroundTruthMessage->covariance[7],submapGroundTruthMessage->covariance[8] );
                        return true;
                }*/

        MapBuilder::~MapBuilder()
        {
            if (submap_blend_vs) glDeleteShader(submap_blend_vs);
            if (submap_blend_fs) glDeleteShader(submap_blend_fs);
            if (submap_blend_sp) glDeleteProgram(submap_blend_sp);
            if (submap_matcher_vs) glDeleteShader(submap_matcher_vs);
            if (submap_matcher_fs) glDeleteShader(submap_matcher_fs);
            if (submap_matcher_sp) glDeleteProgram(submap_matcher_sp);
            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
            glDeleteRenderbuffersEXT(2, map_renderbuffers);
            glDeleteFramebuffersEXT(1, &map_framebuffer);
            cvWaitKey(10); //let window close itself
            cvDestroyAllWindows();
            cvWaitKey(10); //let window close itself

        }
    }
