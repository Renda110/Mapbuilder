namespace mapping {

void LatLonToUtmWGS84( const double LatRad, const double LongRad, double &UTMNorthing, double &UTMEasting, char* UTMZone);
void UtmToLatLonWGS84( const double UTMNorthing, const double UTMEasting, const char* UTMZone, double& LatRad,  double& LongRad );

}
