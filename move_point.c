// utf-8
// gcc -o move_point.so -shared -fPIC move_point.c


// get four pointer of arrays:
// 1. data.shape = (user_len, item_len),  rating data of users.
// 2. matches.shape = (k, user_len),  user-list of k-means.
// 3. matches_len.shape = (k, 1), really length of each user-list.
// 4. points.shape = (k, item_len), centers of each group.

void move_point(int data[], int user_len, int item_len, int k, int matches[], int matches_len[], float points[]){
	float num;
	int sum;
	int rating;
	int i,j,m;
	// for every point
	for(i=0; i<k; i++){
		// for every item
		for(j=0; j<item_len; j++){
			sum = 0;
			num = 0;
			// average rating of group's users
			// for every user
			for(m=0; m<matches_len[i]; m++){
				rating = data[matches[i*user_len + m]*item_len + j];
				sum += rating;
				if(rating != 0) num++;
			}
			if(num==0) points[i*item_len + j] = 0;
			else points[i*item_len + j] = sum/num;
		}
	}
}

