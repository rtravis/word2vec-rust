//  Copyright 2026 Robert Zavalczki
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

/// Compute the square of the euclidean distance between 2 points
fn squared_euclidean_distance(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y)
        .map(|(a, b)| a - b)
        .map(|z| z * z)
        .sum::<f32>()
}

/// Calculate the mean (centroid) of a list of points
fn calculate_mean(vectors: &Vec<&[f32]>, vec_size: usize) -> Vec<f32> {
    let mut accum = vec![0.0_f32; vec_size];
    for vec in vectors {
        debug_assert!(vec.len() == vec_size);
        for (acc, x) in accum.iter_mut().zip(*vec) {
            *acc += x;
        }
    }

    accum.iter_mut().for_each(|x| *x /= vectors.len() as f32);
    accum
}

/// Return points grouped by k-means clustering
/// Uses Lloyd's algorithm, the naive version (https://en.wikipedia.org/wiki/K-means_clustering)
///
/// # Arguments
///
/// * k - number of classes
/// * vectors - the vectors laid out in a continuous memory block of floating points
/// * vec_size - the size of each vector
///
/// # Return Value
///
/// - a vector of k vectors containing the slices representing the points
///
pub fn get_kmeans_clusters(k: usize, vectors: &[f32], vec_size: usize) -> Vec<Vec<&[f32]>> {
    debug_assert!(vectors.len() % vec_size == 0);
    let nvecs = vectors.len() / vec_size;

    if k == 0 {
        panic!("Classes must be strictly positive!");
    }
    if nvecs < k {
        panic!("Classes count is greater than available points!");
    }

    let vec_itr = vectors.chunks_exact(vec_size);

    // Initialize centroids, we should randomly sample, but we just take the first k points
    let mut centroids: Vec<Vec<f32>> = vec![Vec::new(); k];
    for (point, centroid) in vec_itr.zip(centroids.iter_mut()) {
        centroid.extend_from_slice(point);
    }

    loop {
        // Create empty clusters
        let mut clusters = Vec::<Vec<&[f32]>>::new();
        clusters.resize(k, Vec::new());

        // Assign each point to the nearest centroid
        for point in vectors.chunks_exact(vec_size) {
            let mut closest_index: usize = 0;
            let mut min_distance = squared_euclidean_distance(point, &centroids[0]);

            for (idx, cen) in centroids[1..].iter().enumerate() {
                let d = squared_euclidean_distance(point, cen);
                if d < min_distance {
                    min_distance = d;
                    closest_index = idx + 1;
                }
            }
            clusters[closest_index].push(point);
        }

        // Recalculate centroids as the mean of each cluster
        let mut new_centroids: Vec<Vec<f32>> = Vec::new();
        for cluster in &clusters {
            let m = calculate_mean(cluster, vec_size);
            new_centroids.push(m);
        }

        if new_centroids == centroids {
            // algorithm converged, return clusters
            break clusters;
        }
        centroids = new_centroids;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_two_class_2d() {
        let vectors: Vec<f32> = vec![
            -1.5, -1.5, -1.25, -0.75, -2.0, -3.0, 0.95, 1.15, 1.2, 1.3, 0.5, 1.5,
        ];
        let clusters = get_kmeans_clusters(2, &vectors[..], 2);
        assert_eq!(
            clusters[0],
            vec![[-1.5, -1.5], [-1.25, -0.75], [-2.0, -3.0]]
        );
        assert_eq!(clusters[1], vec![[0.95, 1.15], [1.2, 1.3], [0.5, 1.5]]);
    }

    #[test]
    fn kmeans_three_class_2d() {
        let vectors: Vec<f32> = vec![
            -1.5, -1.5, -1.25, -0.75, -2.0, -3.0, 0.95, 1.15, 1.2, 1.3, 0.5, 1.5,
        ];
        let clusters = get_kmeans_clusters(3, &vectors[..], 2);
        assert_eq!(clusters[0], vec![[-1.5, -1.5], [-1.25, -0.75]]);
        assert_eq!(clusters[1], vec![[0.95, 1.15], [1.2, 1.3], [0.5, 1.5]]);
        assert_eq!(clusters[2], vec![[-2.0, -3.0]]);
    }

    #[test]
    fn kmeans_two_class_3d() {
        let vectors: Vec<f32> = vec![
            -1.5, -1.5, -1.25, -0.75, -2.0, -3.0, 0.95, 1.15, 1.2, 1.3, 0.5, 1.5,
        ];
        let clusters = get_kmeans_clusters(2, &vectors[..], 3);
        assert_eq!(clusters[0], vec![[0.95, 1.15, 1.2], [1.3, 0.5, 1.5]]);
        assert_eq!(clusters[1], vec![[-1.5, -1.5, -1.25], [-0.75, -2.0, -3.0]]);
    }
}
