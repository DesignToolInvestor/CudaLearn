#pragma once

#include <string>
#include <io.h>

typedef long long TickCountT;  // may be different on different OSes

TickCountT ReadTicks();
float TicksToSecs(TickCountT ticks);
