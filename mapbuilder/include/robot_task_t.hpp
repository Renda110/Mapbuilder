/** THIS IS AN AUTOMATICALLY GENERATED FILE.  DO NOT MODIFY
 * BY HAND!!
 *
 * Generated by lcm-gen
 **/

#include <lcm/lcm_coretypes.h>

#ifndef __robot_task_t_hpp__
#define __robot_task_t_hpp__



class robot_task_t
{
    public:
        int8_t     task;

    public:
        static const int8_t   TASK_FREEZE = 0;
        static const int8_t   TASK_WAITING = 1;
        static const int8_t   TASK_GOTO_LATLON = 2;
        static const int8_t   TASK_GOTO_LOCAL = 3;
        static const int8_t   TASK_PANORAMA = 4;
        static const int8_t   TASK_TELEOP = 5;
        static const int8_t   TASK_LOCAL_GAMEPAD = 6;
        static const int8_t   TASK_NEUTRALIZE_STATIC = 7;
        static const int8_t   TASK_TRACK_MOBILE = 8;
        static const int8_t   TASK_LOCAL_GAMEPAD_SAFE = 9;
        static const int8_t   TASK_RESTART = 10;
        static const int8_t   TASK_READY_TO_NEUTRALIZE = 11;
        static const int8_t   TASK_COMM_RELAY = 12;

    public:
        inline int encode(void *buf, int offset, int maxlen) const;
        inline int getEncodedSize() const;
        inline int decode(const void *buf, int offset, int maxlen);
        inline static int64_t getHash();
        inline static const char* getTypeName();

        // LCM support functions. Users should not call these
        inline int _encodeNoHash(void *buf, int offset, int maxlen) const;
        inline int _getEncodedSizeNoHash() const;
        inline int _decodeNoHash(const void *buf, int offset, int maxlen);
        inline static int64_t _computeHash(const __lcm_hash_ptr *p);
};

int robot_task_t::encode(void *buf, int offset, int maxlen) const
{
    int pos = 0, tlen;
    int64_t hash = getHash();

    tlen = __int64_t_encode_array(buf, offset + pos, maxlen - pos, &hash, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = this->_encodeNoHash(buf, offset + pos, maxlen - pos);
    if (tlen < 0) return tlen; else pos += tlen;

    return pos;
}

int robot_task_t::decode(const void *buf, int offset, int maxlen)
{
    int pos = 0, thislen;

    int64_t msg_hash;
    thislen = __int64_t_decode_array(buf, offset + pos, maxlen - pos, &msg_hash, 1);
    if (thislen < 0) return thislen; else pos += thislen;
    if (msg_hash != getHash()) return -1;

    thislen = this->_decodeNoHash(buf, offset + pos, maxlen - pos);
    if (thislen < 0) return thislen; else pos += thislen;

    return pos;
}

int robot_task_t::getEncodedSize() const
{
    return 8 + _getEncodedSizeNoHash();
}

int64_t robot_task_t::getHash()
{
    static int64_t hash = _computeHash(NULL);
    return hash;
}

const char* robot_task_t::getTypeName()
{
    return "robot_task_t";
}

int robot_task_t::_encodeNoHash(void *buf, int offset, int maxlen) const
{
    int pos = 0, tlen;

    tlen = __int8_t_encode_array(buf, offset + pos, maxlen - pos, &this->task, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    return pos;
}

int robot_task_t::_decodeNoHash(const void *buf, int offset, int maxlen)
{
    int pos = 0, tlen;

    tlen = __int8_t_decode_array(buf, offset + pos, maxlen - pos, &this->task, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    return pos;
}

int robot_task_t::_getEncodedSizeNoHash() const
{
    int enc_size = 0;
    enc_size += __int8_t_encoded_array_size(NULL, 1);
    return enc_size;
}

int64_t robot_task_t::_computeHash(const __lcm_hash_ptr *)
{
    int64_t hash = 0x90e9a182dedda41eLL;
    return (hash<<1) + ((hash>>63)&1);
}

#endif