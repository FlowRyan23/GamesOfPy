def in_bounds(x, min, max, inclusive=False):
	if inclusive:
		return min <= x <= max
	else:
		return min <= x < max


def in_bounds_nd(pos, mins, maxs, inclusive=False):
	for i in range(len(pos)):
		if not in_bounds(pos[i], mins[i], maxs[i], inclusive=inclusive):
			return False
	return True
