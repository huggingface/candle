use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Position {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

pub const POSITIONS: [Position; 531] = [
    Position {
        position: [0.0, 0.0, 0.0],
    }, // dummy vector because in the original model indices
    // start at 1
    Position {
        position: [40.6266, 28.3457, -1.10804],
    },
    Position {
        position: [40.0714, 30.4443, -1.10804],
    },
    Position {
        position: [40.7155, 31.1438, -1.10804],
    },
    Position {
        position: [42.0257, 30.4443, -1.10804],
    },
    Position {
        position: [43.4692, 28.3457, -1.10804],
    },
    Position {
        position: [37.5425, 28.3457, 14.5117],
    },
    Position {
        position: [37.0303, 30.4443, 14.2938],
    },
    Position {
        position: [37.6244, 31.1438, 14.5466],
    },
    Position {
        position: [38.8331, 30.4443, 15.0609],
    },
    Position {
        position: [40.1647, 28.3457, 15.6274],
    },
    Position {
        position: [29.0859, 28.3457, 27.1468],
    },
    Position {
        position: [28.6917, 30.4443, 26.7527],
    },
    Position {
        position: [29.149, 31.1438, 27.2099],
    },
    Position {
        position: [30.0792, 30.4443, 28.1402],
    },
    Position {
        position: [31.1041, 28.3457, 29.165],
    },
    Position {
        position: [16.4508, 28.3457, 35.6034],
    },
    Position {
        position: [16.2329, 30.4443, 35.0912],
    },
    Position {
        position: [16.4857, 31.1438, 35.6853],
    },
    Position {
        position: [16.9999, 30.4443, 36.894],
    },
    Position {
        position: [17.5665, 28.3457, 38.2256],
    },
    Position {
        position: [0.831025, 28.3457, 38.6876],
    },
    Position {
        position: [0.831025, 30.4443, 38.1324],
    },
    Position {
        position: [0.831025, 31.1438, 38.7764],
    },
    Position {
        position: [0.831025, 30.4443, 40.0866],
    },
    Position {
        position: [0.831025, 28.3457, 41.5301],
    },
    Position {
        position: [-15.868, 28.3457, 35.6034],
    },
    Position {
        position: [-15.0262, 30.4443, 35.0912],
    },
    Position {
        position: [-14.9585, 31.1438, 35.6853],
    },
    Position {
        position: [-15.3547, 30.4443, 36.894],
    },
    Position {
        position: [-15.9044, 28.3457, 38.2256],
    },
    Position {
        position: [-28.3832, 28.3457, 27.1468],
    },
    Position {
        position: [-27.4344, 30.4443, 26.7527],
    },
    Position {
        position: [-27.6068, 31.1438, 27.2099],
    },
    Position {
        position: [-28.4322, 30.4443, 28.1402],
    },
    Position {
        position: [-29.4421, 28.3457, 29.165],
    },
    Position {
        position: [-36.2402, 28.3457, 14.5117],
    },
    Position {
        position: [-35.52, 30.4443, 14.2938],
    },
    Position {
        position: [-36.0073, 31.1438, 14.5466],
    },
    Position {
        position: [-37.1767, 30.4443, 15.0609],
    },
    Position {
        position: [-38.5027, 28.3457, 15.6274],
    },
    Position {
        position: [-38.9646, 28.3457, -1.10804],
    },
    Position {
        position: [-38.4094, 30.4443, -1.10804],
    },
    Position {
        position: [-39.0534, 31.1438, -1.10804],
    },
    Position {
        position: [-40.3636, 30.4443, -1.10804],
    },
    Position {
        position: [-41.8071, 28.3457, -1.10804],
    },
    Position {
        position: [-35.8804, 28.3457, -16.7278],
    },
    Position {
        position: [-35.3683, 30.4443, -16.5099],
    },
    Position {
        position: [-35.9624, 31.1438, -16.7627],
    },
    Position {
        position: [-37.1711, 30.4443, -17.2769],
    },
    Position {
        position: [-38.5027, 28.3457, -17.8435],
    },
    Position {
        position: [-27.4238, 28.3457, -29.3629],
    },
    Position {
        position: [-27.0297, 30.4443, -28.9687],
    },
    Position {
        position: [-27.4869, 31.1438, -29.426],
    },
    Position {
        position: [-28.4172, 30.4443, -30.3562],
    },
    Position {
        position: [-29.4421, 28.3457, -31.3811],
    },
    Position {
        position: [-14.7887, 28.3457, -37.8195],
    },
    Position {
        position: [-14.5708, 30.4443, -37.3073],
    },
    Position {
        position: [-14.8236, 31.1438, -37.9014],
    },
    Position {
        position: [-15.3379, 30.4443, -39.1101],
    },
    Position {
        position: [-15.9044, 28.3457, -40.4417],
    },
    Position {
        position: [0.831025, 28.3457, -40.9036],
    },
    Position {
        position: [0.831025, 30.4443, -40.3484],
    },
    Position {
        position: [0.831025, 31.1438, -40.9925],
    },
    Position {
        position: [0.831025, 30.4443, -42.3027],
    },
    Position {
        position: [0.831025, 28.3457, -43.7462],
    },
    Position {
        position: [16.4508, 28.3457, -37.8195],
    },
    Position {
        position: [16.2329, 30.4443, -37.3073],
    },
    Position {
        position: [16.4857, 31.1438, -37.9014],
    },
    Position {
        position: [16.9999, 30.4443, -39.1101],
    },
    Position {
        position: [17.5665, 28.3457, -40.4417],
    },
    Position {
        position: [29.0859, 28.3457, -29.3629],
    },
    Position {
        position: [28.6917, 30.4443, -28.9687],
    },
    Position {
        position: [29.149, 31.1438, -29.426],
    },
    Position {
        position: [30.0792, 30.4443, -30.3562],
    },
    Position {
        position: [31.1041, 28.3457, -31.3811],
    },
    Position {
        position: [37.5425, 28.3457, -16.7278],
    },
    Position {
        position: [37.0303, 30.4443, -16.5099],
    },
    Position {
        position: [37.6244, 31.1438, -16.7627],
    },
    Position {
        position: [38.8331, 30.4443, -17.2769],
    },
    Position {
        position: [40.1647, 28.3457, -17.8435],
    },
    Position {
        position: [48.6879, 17.1865, -1.10804],
    },
    Position {
        position: [53.2404, 6.22714, -1.10804],
    },
    Position {
        position: [56.4605, -4.33246, -1.10804],
    },
    Position {
        position: [57.6819, -14.2925, -1.10804],
    },
    Position {
        position: [44.979, 17.1865, 17.6758],
    },
    Position {
        position: [49.1787, 6.22714, 19.4626],
    },
    Position {
        position: [52.1492, -4.33246, 20.7265],
    },
    Position {
        position: [53.2759, -14.2925, 21.2059],
    },
    Position {
        position: [34.8094, 17.1865, 32.8703],
    },
    Position {
        position: [38.0417, 6.22714, 36.1026],
    },
    Position {
        position: [40.3279, -4.33246, 38.3889],
    },
    Position {
        position: [41.1951, -14.2925, 39.2561],
    },
    Position {
        position: [19.6148, 17.1865, 43.0399],
    },
    Position {
        position: [21.4017, 6.22714, 47.2396],
    },
    Position {
        position: [22.6656, -4.33246, 50.2101],
    },
    Position {
        position: [23.145, -14.2925, 51.3369],
    },
    Position {
        position: [0.831025, 17.1865, 46.7488],
    },
    Position {
        position: [0.831025, 6.22714, 51.3013],
    },
    Position {
        position: [0.831025, -4.33246, 54.5214],
    },
    Position {
        position: [0.831025, -14.2925, 55.7428],
    },
    Position {
        position: [-17.9528, 17.1865, 43.0399],
    },
    Position {
        position: [-19.7397, 6.22714, 47.2396],
    },
    Position {
        position: [-21.0035, -4.33246, 50.2101],
    },
    Position {
        position: [-21.4829, -14.2925, 51.3369],
    },
    Position {
        position: [-33.1474, 17.1865, 32.8703],
    },
    Position {
        position: [-36.3796, 6.22714, 36.1026],
    },
    Position {
        position: [-38.6659, -4.33246, 38.3889],
    },
    Position {
        position: [-39.5331, -14.2925, 39.2561],
    },
    Position {
        position: [-43.3169, 17.1865, 17.6758],
    },
    Position {
        position: [-47.5166, 6.22714, 19.4626],
    },
    Position {
        position: [-50.4871, -4.33246, 20.7265],
    },
    Position {
        position: [-51.6139, -14.2925, 21.2059],
    },
    Position {
        position: [-47.0258, 17.1865, -1.10804],
    },
    Position {
        position: [-51.5784, 6.22714, -1.10804],
    },
    Position {
        position: [-54.7984, -4.33246, -1.10804],
    },
    Position {
        position: [-56.0198, -14.2925, -1.10804],
    },
    Position {
        position: [-43.3169, 17.1865, -19.8919],
    },
    Position {
        position: [-47.5166, 6.22714, -21.6787],
    },
    Position {
        position: [-50.4871, -4.33246, -22.9426],
    },
    Position {
        position: [-51.6139, -14.2925, -23.422],
    },
    Position {
        position: [-33.1474, 17.1865, -35.0864],
    },
    Position {
        position: [-36.3796, 6.22714, -38.3187],
    },
    Position {
        position: [-38.6659, -4.33246, -40.6049],
    },
    Position {
        position: [-39.5331, -14.2925, -41.4721],
    },
    Position {
        position: [-17.9528, 17.1865, -45.256],
    },
    Position {
        position: [-19.7397, 6.22714, -49.4557],
    },
    Position {
        position: [-21.0035, -4.33246, -52.4262],
    },
    Position {
        position: [-21.4829, -14.2925, -53.5529],
    },
    Position {
        position: [0.831025, 17.1865, -48.9649],
    },
    Position {
        position: [0.831025, 6.22714, -53.5174],
    },
    Position {
        position: [0.831025, -4.33246, -56.7375],
    },
    Position {
        position: [0.831025, -14.2925, -57.9589],
    },
    Position {
        position: [19.6148, 17.1865, -45.256],
    },
    Position {
        position: [21.4017, 6.22714, -49.4557],
    },
    Position {
        position: [22.6656, -4.33246, -52.4262],
    },
    Position {
        position: [23.145, -14.2925, -53.5529],
    },
    Position {
        position: [34.8094, 17.1865, -35.0864],
    },
    Position {
        position: [38.0417, 6.22714, -38.3187],
    },
    Position {
        position: [40.3279, -4.33246, -40.6049],
    },
    Position {
        position: [41.1951, -14.2925, -41.4721],
    },
    Position {
        position: [44.979, 17.1865, -19.8919],
    },
    Position {
        position: [49.1787, 6.22714, -21.6787],
    },
    Position {
        position: [52.1492, -4.33246, -22.9426],
    },
    Position {
        position: [53.2759, -14.2925, -23.422],
    },
    Position {
        position: [55.4611, -22.7202, -1.10804],
    },
    Position {
        position: [50.5755, -28.9493, -1.10804],
    },
    Position {
        position: [45.6899, -33.1798, -1.10804],
    },
    Position {
        position: [43.4692, -35.6115, -1.10804],
    },
    Position {
        position: [51.2273, -22.7202, 20.3343],
    },
    Position {
        position: [46.7203, -28.9493, 18.4167],
    },
    Position {
        position: [42.2133, -33.1798, 16.4991],
    },
    Position {
        position: [40.1647, -35.6115, 15.6274],
    },
    Position {
        position: [39.6184, -22.7202, 37.6793],
    },
    Position {
        position: [36.1496, -28.9493, 34.2106],
    },
    Position {
        position: [32.6808, -33.1798, 30.7418],
    },
    Position {
        position: [31.1041, -35.6115, 29.165],
    },
    Position {
        position: [22.2733, -22.7202, 49.2882],
    },
    Position {
        position: [20.3557, -28.9493, 44.7813],
    },
    Position {
        position: [18.4381, -33.1798, 40.2743],
    },
    Position {
        position: [17.5665, -35.6115, 38.2256],
    },
    Position {
        position: [0.831025, -22.7202, 53.5221],
    },
    Position {
        position: [0.831025, -28.9493, 48.6365],
    },
    Position {
        position: [0.831025, -33.1798, 43.7508],
    },
    Position {
        position: [0.831025, -35.6115, 41.5301],
    },
    Position {
        position: [-20.6113, -22.7202, 49.2882],
    },
    Position {
        position: [-18.6937, -28.9493, 44.7813],
    },
    Position {
        position: [-16.7761, -33.1798, 40.2743],
    },
    Position {
        position: [-15.9044, -35.6115, 38.2256],
    },
    Position {
        position: [-37.9564, -22.7202, 37.6793],
    },
    Position {
        position: [-34.4876, -28.9493, 34.2106],
    },
    Position {
        position: [-31.0188, -33.1798, 30.7418],
    },
    Position {
        position: [-29.4421, -35.6115, 29.165],
    },
    Position {
        position: [-49.5653, -22.7202, 20.3343],
    },
    Position {
        position: [-45.0583, -28.9493, 18.4167],
    },
    Position {
        position: [-40.5513, -33.1798, 16.4991],
    },
    Position {
        position: [-38.5027, -35.6115, 15.6274],
    },
    Position {
        position: [-53.7991, -22.7202, -1.10804],
    },
    Position {
        position: [-48.9135, -28.9493, -1.10804],
    },
    Position {
        position: [-44.0279, -33.1798, -1.10804],
    },
    Position {
        position: [-41.8071, -35.6115, -1.10804],
    },
    Position {
        position: [-49.5653, -22.7202, -22.5504],
    },
    Position {
        position: [-45.0583, -28.9493, -20.6327],
    },
    Position {
        position: [-40.5513, -33.1798, -18.7151],
    },
    Position {
        position: [-38.5027, -35.6115, -17.8435],
    },
    Position {
        position: [-37.9564, -22.7202, -39.8954],
    },
    Position {
        position: [-34.4876, -28.9493, -36.4266],
    },
    Position {
        position: [-31.0188, -33.1798, -32.9578],
    },
    Position {
        position: [-29.4421, -35.6115, -31.3811],
    },
    Position {
        position: [-20.6113, -22.7202, -51.5043],
    },
    Position {
        position: [-18.6937, -28.9493, -46.9973],
    },
    Position {
        position: [-16.7761, -33.1798, -42.4903],
    },
    Position {
        position: [-15.9044, -35.6115, -40.4417],
    },
    Position {
        position: [0.831025, -22.7202, -55.7382],
    },
    Position {
        position: [0.831025, -28.9493, -50.8525],
    },
    Position {
        position: [0.831025, -33.1798, -45.9669],
    },
    Position {
        position: [0.831025, -35.6115, -43.7462],
    },
    Position {
        position: [22.2733, -22.7202, -51.5043],
    },
    Position {
        position: [20.3557, -28.9493, -46.9973],
    },
    Position {
        position: [18.4381, -33.1798, -42.4903],
    },
    Position {
        position: [17.5665, -35.6115, -40.4417],
    },
    Position {
        position: [39.6184, -22.7202, -39.8954],
    },
    Position {
        position: [36.1496, -28.9493, -36.4266],
    },
    Position {
        position: [32.6808, -33.1798, -32.9578],
    },
    Position {
        position: [31.1041, -35.6115, -31.3811],
    },
    Position {
        position: [51.2273, -22.7202, -22.5504],
    },
    Position {
        position: [46.7203, -28.9493, -20.6327],
    },
    Position {
        position: [42.2133, -33.1798, -18.7151],
    },
    Position {
        position: [40.1647, -35.6115, -17.8435],
    },
    Position {
        position: [42.5031, -37.1772, -1.10804],
    },
    Position {
        position: [37.3399, -38.5429, -1.10804],
    },
    Position {
        position: [24.5818, -39.5089, -1.10804],
    },
    Position {
        position: [0.831025, -39.8754, -1.10804],
    },
    Position {
        position: [39.2736, -37.1772, 15.2483],
    },
    Position {
        position: [34.5105, -38.5429, 13.2217],
    },
    Position {
        position: [22.7411, -39.5089, 8.21414],
    },
    Position {
        position: [30.4182, -37.1772, 28.4792],
    },
    Position {
        position: [26.7523, -38.5429, 24.8133],
    },
    Position {
        position: [17.6941, -39.5089, 15.755],
    },
    Position {
        position: [17.1873, -37.1772, 37.3345],
    },
    Position {
        position: [15.1608, -38.5429, 32.5714],
    },
    Position {
        position: [10.1532, -39.5089, 20.8021],
    },
    Position {
        position: [0.831025, -37.1772, 40.5641],
    },
    Position {
        position: [0.831025, -38.5429, 35.4009],
    },
    Position {
        position: [0.831025, -39.5089, 22.6427],
    },
    Position {
        position: [-15.5253, -37.1772, 37.3345],
    },
    Position {
        position: [-13.4987, -38.5429, 32.5714],
    },
    Position {
        position: [-8.49115, -39.5089, 20.8021],
    },
    Position {
        position: [-28.7562, -37.1772, 28.4792],
    },
    Position {
        position: [-25.0903, -38.5429, 24.8133],
    },
    Position {
        position: [-16.032, -39.5089, 15.755],
    },
    Position {
        position: [-37.6115, -37.1772, 15.2483],
    },
    Position {
        position: [-32.8484, -38.5429, 13.2217],
    },
    Position {
        position: [-21.0791, -39.5089, 8.21414],
    },
    Position {
        position: [-40.8411, -37.1772, -1.10804],
    },
    Position {
        position: [-35.6779, -38.5429, -1.10804],
    },
    Position {
        position: [-22.9198, -39.5089, -1.10804],
    },
    Position {
        position: [-37.6115, -37.1772, -17.4643],
    },
    Position {
        position: [-32.8484, -38.5429, -15.4378],
    },
    Position {
        position: [-21.0791, -39.5089, -10.4302],
    },
    Position {
        position: [-28.7562, -37.1772, -30.6952],
    },
    Position {
        position: [-25.0903, -38.5429, -27.0294],
    },
    Position {
        position: [-16.032, -39.5089, -17.9711],
    },
    Position {
        position: [-15.5253, -37.1772, -39.5506],
    },
    Position {
        position: [-13.4987, -38.5429, -34.7875],
    },
    Position {
        position: [-8.49115, -39.5089, -23.0181],
    },
    Position {
        position: [0.831025, -37.1772, -42.7802],
    },
    Position {
        position: [0.831025, -38.5429, -37.6169],
    },
    Position {
        position: [0.831025, -39.5089, -24.8588],
    },
    Position {
        position: [17.1873, -37.1772, -39.5506],
    },
    Position {
        position: [15.1608, -38.5429, -34.7875],
    },
    Position {
        position: [10.1532, -39.5089, -23.0181],
    },
    Position {
        position: [30.4182, -37.1772, -30.6952],
    },
    Position {
        position: [26.7523, -38.5429, -27.0294],
    },
    Position {
        position: [17.6941, -39.5089, -17.9711],
    },
    Position {
        position: [39.2736, -37.1772, -17.4643],
    },
    Position {
        position: [34.5105, -38.5429, -15.4378],
    },
    Position {
        position: [22.7411, -39.5089, -10.4302],
    },
    Position {
        position: [-44.6497, 17.6861, -1.10804],
    },
    Position {
        position: [-57.9297, 17.5862, -1.10804],
    },
    Position {
        position: [-67.7453, 16.8867, -1.10804],
    },
    Position {
        position: [-73.8301, 14.9879, -1.10804],
    },
    Position {
        position: [-75.9176, 11.2904, -1.10804],
    },
    Position {
        position: [-44.2055, 18.6855, 3.68876],
    },
    Position {
        position: [-58.3252, 18.5699, 3.68876],
    },
    Position {
        position: [-68.6891, 17.7611, 3.68876],
    },
    Position {
        position: [-75.0724, 15.5657, 3.68876],
    },
    Position {
        position: [-77.2501, 11.2904, 3.68876],
    },
    Position {
        position: [-43.2284, 20.884, 5.28769],
    },
    Position {
        position: [-59.1955, 20.7341, 5.28769],
    },
    Position {
        position: [-70.7655, 19.6848, 5.28769],
    },
    Position {
        position: [-77.8053, 16.8367, 5.28769],
    },
    Position {
        position: [-80.1814, 11.2904, 5.28769],
    },
    Position {
        position: [-42.2513, 23.0825, 3.68876],
    },
    Position {
        position: [-60.0657, 22.8983, 3.68876],
    },
    Position {
        position: [-72.8419, 21.6085, 3.68876],
    },
    Position {
        position: [-80.5381, 18.1077, 3.68876],
    },
    Position {
        position: [-83.1128, 11.2904, 3.68876],
    },
    Position {
        position: [-41.8071, 24.0819, -1.10804],
    },
    Position {
        position: [-60.4613, 23.882, -1.10804],
    },
    Position {
        position: [-73.7857, 22.4829, -1.10804],
    },
    Position {
        position: [-81.7804, 18.6855, -1.10804],
    },
    Position {
        position: [-84.4453, 11.2904, -1.10804],
    },
    Position {
        position: [-42.2513, 23.0825, -5.90483],
    },
    Position {
        position: [-60.0657, 22.8983, -5.90483],
    },
    Position {
        position: [-72.8419, 21.6085, -5.90483],
    },
    Position {
        position: [-80.5381, 18.1077, -5.90483],
    },
    Position {
        position: [-83.1128, 11.2904, -5.90483],
    },
    Position {
        position: [-43.2284, 20.884, -7.50376],
    },
    Position {
        position: [-59.1955, 20.7341, -7.50376],
    },
    Position {
        position: [-70.7655, 19.6848, -7.50376],
    },
    Position {
        position: [-77.8053, 16.8367, -7.50376],
    },
    Position {
        position: [-80.1814, 11.2904, -7.50376],
    },
    Position {
        position: [-44.2055, 18.6855, -5.90483],
    },
    Position {
        position: [-58.3252, 18.5699, -5.90483],
    },
    Position {
        position: [-68.6891, 17.7611, -5.90483],
    },
    Position {
        position: [-75.0724, 15.5657, -5.90483],
    },
    Position {
        position: [-77.2501, 11.2904, -5.90483],
    },
    Position {
        position: [-74.8073, 5.4943, -1.10804],
    },
    Position {
        position: [-71.2985, -1.50103, -1.10804],
    },
    Position {
        position: [-65.1248, -8.49634, -1.10804],
    },
    Position {
        position: [-56.0198, -14.2925, -1.10804],
    },
    Position {
        position: [-76.0183, 4.93477, 3.68876],
    },
    Position {
        position: [-72.159, -2.35462, 3.68876],
    },
    Position {
        position: [-65.4267, -9.55033, 3.68876],
    },
    Position {
        position: [-55.5757, -15.6249, 3.68876],
    },
    Position {
        position: [-78.6824, 3.70383, 5.28769],
    },
    Position {
        position: [-74.0522, -4.23253, 5.28769],
    },
    Position {
        position: [-66.0909, -11.8691, 5.28769],
    },
    Position {
        position: [-54.5986, -18.5563, 5.28769],
    },
    Position {
        position: [-81.3466, 2.47288, 3.68876],
    },
    Position {
        position: [-75.9454, -6.11044, 3.68876],
    },
    Position {
        position: [-66.755, -14.1878, 3.68876],
    },
    Position {
        position: [-53.6214, -21.4877, 3.68876],
    },
    Position {
        position: [-82.5576, 1.91336, -1.10804],
    },
    Position {
        position: [-76.8059, -6.96404, -1.10804],
    },
    Position {
        position: [-67.0569, -15.2418, -1.10804],
    },
    Position {
        position: [-53.1773, -22.8201, -1.10804],
    },
    Position {
        position: [-81.3466, 2.47288, -5.90483],
    },
    Position {
        position: [-75.9454, -6.11044, -5.90483],
    },
    Position {
        position: [-66.755, -14.1878, -5.90483],
    },
    Position {
        position: [-53.6214, -21.4877, -5.90483],
    },
    Position {
        position: [-78.6824, 3.70383, -7.50376],
    },
    Position {
        position: [-74.0522, -4.23253, -7.50376],
    },
    Position {
        position: [-66.0909, -11.8691, -7.50376],
    },
    Position {
        position: [-54.5986, -18.5563, -7.50376],
    },
    Position {
        position: [-76.0183, 4.93477, -5.90483],
    },
    Position {
        position: [-72.159, -2.35462, -5.90483],
    },
    Position {
        position: [-65.4267, -9.55033, -5.90483],
    },
    Position {
        position: [-55.5757, -15.6249, -5.90483],
    },
    Position {
        position: [49.1543, 0.630882, -1.10804],
    },
    Position {
        position: [62.7896, 3.76212, -1.10804],
    },
    Position {
        position: [68.6967, 11.2904, -1.10804],
    },
    Position {
        position: [71.939, 20.4176, -1.10804],
    },
    Position {
        position: [77.5797, 28.3457, -1.10804],
    },
    Position {
        position: [49.1543, -3.03333, 9.4449],
    },
    Position {
        position: [63.8305, 1.04519, 8.42059],
    },
    Position {
        position: [70.0292, 9.70814, 6.1671],
    },
    Position {
        position: [73.5629, 19.8451, 3.91361],
    },
    Position {
        position: [80.2446, 28.3457, 2.88929],
    },
    Position {
        position: [49.1543, -11.0946, 12.9626],
    },
    Position {
        position: [66.1207, -4.93206, 11.5968],
    },
    Position {
        position: [72.9605, 6.22714, 8.59214],
    },
    Position {
        position: [77.1355, 18.5855, 5.58749],
    },
    Position {
        position: [86.1073, 28.3457, 4.22173],
    },
    Position {
        position: [49.1543, -19.1559, 9.4449],
    },
    Position {
        position: [68.4108, -10.9093, 8.42059],
    },
    Position {
        position: [75.8919, 2.74614, 6.1671],
    },
    Position {
        position: [80.7081, 17.326, 3.91361],
    },
    Position {
        position: [91.97, 28.3457, 2.88929],
    },
    Position {
        position: [49.1543, -22.8201, -1.10804],
    },
    Position {
        position: [69.4518, -13.6262, -1.10804],
    },
    Position {
        position: [77.2244, 1.16386, -1.10804],
    },
    Position {
        position: [82.3321, 16.7534, -1.10804],
    },
    Position {
        position: [94.6349, 28.3457, -1.10804],
    },
    Position {
        position: [49.1543, -19.1559, -11.661],
    },
    Position {
        position: [68.4108, -10.9093, -10.6367],
    },
    Position {
        position: [75.8919, 2.74614, -8.38317],
    },
    Position {
        position: [80.7081, 17.326, -6.12968],
    },
    Position {
        position: [91.97, 28.3457, -5.10536],
    },
    Position {
        position: [49.1543, -11.0946, -15.1786],
    },
    Position {
        position: [66.1207, -4.93206, -13.8129],
    },
    Position {
        position: [72.9605, 6.22714, -10.8082],
    },
    Position {
        position: [77.1355, 18.5855, -7.80356],
    },
    Position {
        position: [86.1073, 28.3457, -6.4378],
    },
    Position {
        position: [49.1543, -3.03333, -11.661],
    },
    Position {
        position: [63.8305, 1.04519, -10.6367],
    },
    Position {
        position: [70.0292, 9.70814, -8.38317],
    },
    Position {
        position: [73.5629, 19.8451, -6.12968],
    },
    Position {
        position: [80.2446, 28.3457, -5.10536],
    },
    Position {
        position: [79.6227, 29.5449, -1.10804],
    },
    Position {
        position: [81.1329, 29.9446, -1.10804],
    },
    Position {
        position: [81.577, 29.5449, -1.10804],
    },
    Position {
        position: [80.4222, 28.3457, -1.10804],
    },
    Position {
        position: [82.4767, 29.6034, 2.63946],
    },
    Position {
        position: [83.8116, 30.0383, 2.08983],
    },
    Position {
        position: [83.8515, 29.6268, 1.54019],
    },
    Position {
        position: [82.1988, 28.3457, 1.29036],
    },
    Position {
        position: [88.7555, 29.7322, 3.88862],
    },
    Position {
        position: [89.7049, 30.2444, 3.15578],
    },
    Position {
        position: [88.8555, 29.8072, 2.42294],
    },
    Position {
        position: [86.1073, 28.3457, 2.08983],
    },
    Position {
        position: [95.0343, 29.8611, 2.63946],
    },
    Position {
        position: [95.5982, 30.4505, 2.08983],
    },
    Position {
        position: [93.8594, 29.9875, 1.54019],
    },
    Position {
        position: [90.0158, 28.3457, 1.29036],
    },
    Position {
        position: [97.8883, 29.9196, -1.10804],
    },
    Position {
        position: [98.2769, 30.5442, -1.10804],
    },
    Position {
        position: [96.1339, 30.0695, -1.10804],
    },
    Position {
        position: [91.7924, 28.3457, -1.10804],
    },
    Position {
        position: [95.0343, 29.8611, -4.85553],
    },
    Position {
        position: [95.5982, 30.4505, -4.3059],
    },
    Position {
        position: [93.8594, 29.9875, -3.75626],
    },
    Position {
        position: [90.0158, 28.3457, -3.50643],
    },
    Position {
        position: [88.7555, 29.7322, -6.10469],
    },
    Position {
        position: [89.7049, 30.2444, -5.37185],
    },
    Position {
        position: [88.8555, 29.8072, -4.63901],
    },
    Position {
        position: [86.1073, 28.3457, -4.3059],
    },
    Position {
        position: [82.4767, 29.6034, -4.85553],
    },
    Position {
        position: [83.8116, 30.0383, -4.3059],
    },
    Position {
        position: [83.8515, 29.6268, -3.75626],
    },
    Position {
        position: [82.1988, 28.3457, -3.50643],
    },
    Position {
        position: [0.831025, 49.6647, -1.10804],
    },
    Position {
        position: [10.5134, 48.2657, -1.10804],
    },
    Position {
        position: [10.0693, 44.868, -1.10804],
    },
    Position {
        position: [6.42728, 40.6708, -1.10804],
    },
    Position {
        position: [6.51611, 36.8733, -1.10804],
    },
    Position {
        position: [9.76642, 48.2657, 2.70243],
    },
    Position {
        position: [9.35632, 44.868, 2.52698],
    },
    Position {
        position: [5.9947, 40.6708, 1.09187],
    },
    Position {
        position: [6.07552, 36.8733, 1.12336],
    },
    Position {
        position: [7.71453, 48.2657, 5.77547],
    },
    Position {
        position: [7.39819, 44.868, 5.45913],
    },
    Position {
        position: [4.80736, 40.6708, 2.8683],
    },
    Position {
        position: [4.86744, 36.8733, 2.92838],
    },
    Position {
        position: [4.64149, 48.2657, 7.82736],
    },
    Position {
        position: [4.46604, 44.868, 7.41726],
    },
    Position {
        position: [3.03093, 40.6708, 4.05564],
    },
    Position {
        position: [3.06242, 36.8733, 4.13646],
    },
    Position {
        position: [0.831025, 48.2657, 8.57438],
    },
    Position {
        position: [0.831025, 44.868, 8.13023],
    },
    Position {
        position: [0.831025, 40.6708, 4.48822],
    },
    Position {
        position: [0.831025, 36.8733, 4.57705],
    },
    Position {
        position: [-2.97944, 48.2657, 7.82736],
    },
    Position {
        position: [-2.80399, 44.868, 7.41726],
    },
    Position {
        position: [-1.36888, 40.6708, 4.05564],
    },
    Position {
        position: [-1.40037, 36.8733, 4.13646],
    },
    Position {
        position: [-6.05248, 48.2657, 5.77547],
    },
    Position {
        position: [-5.73614, 44.868, 5.45913],
    },
    Position {
        position: [-3.14531, 40.6708, 2.8683],
    },
    Position {
        position: [-3.20539, 36.8733, 2.92838],
    },
    Position {
        position: [-8.10437, 48.2657, 2.70243],
    },
    Position {
        position: [-7.69427, 44.868, 2.52698],
    },
    Position {
        position: [-4.33265, 40.6708, 1.09187],
    },
    Position {
        position: [-4.41347, 36.8733, 1.12336],
    },
    Position {
        position: [-8.85139, 48.2657, -1.10804],
    },
    Position {
        position: [-8.40724, 44.868, -1.10804],
    },
    Position {
        position: [-4.76523, 40.6708, -1.10804],
    },
    Position {
        position: [-4.85406, 36.8733, -1.10804],
    },
    Position {
        position: [-8.10437, 48.2657, -4.9185],
    },
    Position {
        position: [-7.69427, 44.868, -4.74305],
    },
    Position {
        position: [-4.33265, 40.6708, -3.30794],
    },
    Position {
        position: [-4.41347, 36.8733, -3.33943],
    },
    Position {
        position: [-6.05248, 48.2657, -7.99154],
    },
    Position {
        position: [-5.73614, 44.868, -7.6752],
    },
    Position {
        position: [-3.14531, 40.6708, -5.08437],
    },
    Position {
        position: [-3.20539, 36.8733, -5.14445],
    },
    Position {
        position: [-2.97944, 48.2657, -10.0434],
    },
    Position {
        position: [-2.80399, 44.868, -9.63333],
    },
    Position {
        position: [-1.36888, 40.6708, -6.27171],
    },
    Position {
        position: [-1.40037, 36.8733, -6.35253],
    },
    Position {
        position: [0.831025, 48.2657, -10.7904],
    },
    Position {
        position: [0.831025, 44.868, -10.3463],
    },
    Position {
        position: [0.831025, 40.6708, -6.70429],
    },
    Position {
        position: [0.831025, 36.8733, -6.79312],
    },
    Position {
        position: [4.64149, 48.2657, -10.0434],
    },
    Position {
        position: [4.46604, 44.868, -9.63333],
    },
    Position {
        position: [3.03093, 40.6708, -6.27171],
    },
    Position {
        position: [3.06242, 36.8733, -6.35253],
    },
    Position {
        position: [7.71453, 48.2657, -7.99154],
    },
    Position {
        position: [7.39819, 44.868, -7.6752],
    },
    Position {
        position: [4.80736, 40.6708, -5.08437],
    },
    Position {
        position: [4.86744, 36.8733, -5.14445],
    },
    Position {
        position: [9.76642, 48.2657, -4.9185],
    },
    Position {
        position: [9.35632, 44.868, -4.74305],
    },
    Position {
        position: [5.9947, 40.6708, -3.30794],
    },
    Position {
        position: [6.07552, 36.8733, -3.33943],
    },
    Position {
        position: [13.8001, 34.3417, -1.10804],
    },
    Position {
        position: [24.282, 32.6095, -1.10804],
    },
    Position {
        position: [33.6979, 30.8773, -1.10804],
    },
    Position {
        position: [37.7841, 28.3457, -1.10804],
    },
    Position {
        position: [12.795, 34.3417, 3.98234],
    },
    Position {
        position: [22.4646, 32.6095, 8.09647],
    },
    Position {
        position: [31.1507, 30.8773, 11.7922],
    },
    Position {
        position: [34.9202, 28.3457, 13.396],
    },
    Position {
        position: [10.0391, 34.3417, 8.10003],
    },
    Position {
        position: [17.4812, 32.6095, 15.5422],
    },
    Position {
        position: [24.1665, 30.8773, 22.2275],
    },
    Position {
        position: [27.0677, 28.3457, 25.1286],
    },
    Position {
        position: [5.9214, 34.3417, 10.856],
    },
    Position {
        position: [10.0355, 32.6095, 20.5255],
    },
    Position {
        position: [13.7313, 30.8773, 29.2117],
    },
    Position {
        position: [15.3351, 28.3457, 32.9812],
    },
    Position {
        position: [0.831025, 34.3417, 11.8611],
    },
    Position {
        position: [0.831025, 32.6095, 22.3429],
    },
    Position {
        position: [0.831025, 30.8773, 31.7589],
    },
    Position {
        position: [0.831025, 28.3457, 35.845],
    },
    Position {
        position: [-4.25935, 34.3417, 10.856],
    },
    Position {
        position: [-8.37348, 32.6095, 20.5255],
    },
    Position {
        position: [-12.0692, 30.8773, 29.2117],
    },
    Position {
        position: [-13.673, 28.3457, 32.9812],
    },
    Position {
        position: [-8.37704, 34.3417, 8.10003],
    },
    Position {
        position: [-15.8192, 32.6095, 15.5422],
    },
    Position {
        position: [-22.5045, 30.8773, 22.2275],
    },
    Position {
        position: [-25.4056, 28.3457, 25.1286],
    },
    Position {
        position: [-11.133, 34.3417, 3.98234],
    },
    Position {
        position: [-20.8025, 32.6095, 8.09647],
    },
    Position {
        position: [-29.4887, 30.8773, 11.7922],
    },
    Position {
        position: [-33.2582, 28.3457, 13.396],
    },
    Position {
        position: [-12.1381, 34.3417, -1.10804],
    },
    Position {
        position: [-22.62, 32.6095, -1.10804],
    },
    Position {
        position: [-32.0359, 30.8773, -1.10804],
    },
    Position {
        position: [-36.122, 28.3457, -1.10804],
    },
    Position {
        position: [-11.133, 34.3417, -6.19841],
    },
    Position {
        position: [-20.8025, 32.6095, -10.3125],
    },
    Position {
        position: [-29.4887, 30.8773, -14.0083],
    },
    Position {
        position: [-33.2582, 28.3457, -15.6121],
    },
    Position {
        position: [-8.37704, 34.3417, -10.3161],
    },
    Position {
        position: [-15.8192, 32.6095, -17.7582],
    },
    Position {
        position: [-22.5045, 30.8773, -24.4435],
    },
    Position {
        position: [-25.4056, 28.3457, -27.3447],
    },
    Position {
        position: [-4.25935, 34.3417, -13.072],
    },
    Position {
        position: [-8.37348, 32.6095, -22.7416],
    },
    Position {
        position: [-12.0692, 30.8773, -31.4277],
    },
    Position {
        position: [-13.673, 28.3457, -35.1972],
    },
    Position {
        position: [0.831025, 34.3417, -14.0771],
    },
    Position {
        position: [0.831025, 32.6095, -24.559],
    },
    Position {
        position: [0.831025, 30.8773, -33.9749],
    },
    Position {
        position: [0.831025, 28.3457, -38.0611],
    },
    Position {
        position: [5.9214, 34.3417, -13.072],
    },
    Position {
        position: [10.0355, 32.6095, -22.7416],
    },
    Position {
        position: [13.7313, 30.8773, -31.4277],
    },
    Position {
        position: [15.3351, 28.3457, -35.1972],
    },
    Position {
        position: [10.0391, 34.3417, -10.3161],
    },
    Position {
        position: [17.4812, 32.6095, -17.7582],
    },
    Position {
        position: [24.1665, 30.8773, -24.4435],
    },
    Position {
        position: [27.0677, 28.3457, -27.3447],
    },
    Position {
        position: [12.795, 34.3417, -6.19841],
    },
    Position {
        position: [22.4646, 32.6095, -10.3125],
    },
    Position {
        position: [31.1507, 30.8773, -14.0083],
    },
    Position {
        position: [34.9202, 28.3457, -15.6121],
    },
];

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Normal {
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
}

pub const NORMALS: [Normal; 531] = [
    Normal {
        normal: [0.0, 0.0, 0.0],
    }, // dummy vector because in the original model indices
    // start at 1
    Normal {
        normal: [-0.966742, -0.255752, 0.0],
    },
    Normal {
        normal: [-0.966824, 0.255443, 0.0],
    },
    Normal {
        normal: [-0.092052, 0.995754, 0.0],
    },
    Normal {
        normal: [0.68205, 0.731305, 0.0],
    },
    Normal {
        normal: [0.870301, 0.492521, -0.0],
    },
    Normal {
        normal: [-0.893014, -0.256345, -0.369882],
    },
    Normal {
        normal: [-0.893437, 0.255997, -0.369102],
    },
    Normal {
        normal: [-0.0838771, 0.995843, -0.0355068],
    },
    Normal {
        normal: [0.629724, 0.73186, 0.260439],
    },
    Normal {
        normal: [0.803725, 0.49337, 0.332584],
    },
    Normal {
        normal: [-0.683407, -0.256729, -0.683407],
    },
    Normal {
        normal: [-0.683531, 0.256067, -0.683531],
    },
    Normal {
        normal: [-0.0649249, 0.995776, -0.0649248],
    },
    Normal {
        normal: [0.481398, 0.732469, 0.481398],
    },
    Normal {
        normal: [0.614804, 0.493997, 0.614804],
    },
    Normal {
        normal: [-0.369882, -0.256345, -0.893014],
    },
    Normal {
        normal: [-0.369102, 0.255997, -0.893437],
    },
    Normal {
        normal: [-0.0355067, 0.995843, -0.0838772],
    },
    Normal {
        normal: [0.260439, 0.73186, 0.629724],
    },
    Normal {
        normal: [0.332584, 0.49337, 0.803725],
    },
    Normal {
        normal: [-0.00284834, -0.257863, -0.966177],
    },
    Normal {
        normal: [-0.00192311, 0.254736, -0.967009],
    },
    Normal {
        normal: [-0.000266114, 0.995734, -0.0922702],
    },
    Normal {
        normal: [0.0, 0.731295, 0.682061],
    },
    Normal {
        normal: [0.0, 0.492521, 0.870301],
    },
    Normal {
        normal: [0.379058, -0.3593, -0.852771],
    },
    Normal {
        normal: [0.37711, 0.149086, -0.914091],
    },
    Normal {
        normal: [0.0275022, 0.992081, -0.122551],
    },
    Normal {
        normal: [-0.26101, 0.726762, 0.635367],
    },
    Normal {
        normal: [-0.332485, 0.492546, 0.804271],
    },
    Normal {
        normal: [0.663548, -0.410791, -0.625264],
    },
    Normal {
        normal: [0.712664, 0.0737216, -0.697621],
    },
    Normal {
        normal: [0.0997268, 0.987509, -0.121984],
    },
    Normal {
        normal: [-0.48732, 0.723754, 0.488568],
    },
    Normal {
        normal: [-0.615242, 0.492602, 0.615484],
    },
    Normal {
        normal: [0.880028, -0.332908, -0.338709],
    },
    Normal {
        normal: [0.917276, 0.167113, -0.361493],
    },
    Normal {
        normal: [0.113584, 0.992365, -0.0480695],
    },
    Normal {
        normal: [-0.63415, 0.727508, 0.261889],
    },
    Normal {
        normal: [-0.804126, 0.492634, 0.332705],
    },
    Normal {
        normal: [0.96669, -0.255738, 0.0104537],
    },
    Normal {
        normal: [0.967442, 0.252962, 0.00810329],
    },
    Normal {
        normal: [0.0934365, 0.995624, 0.00128063],
    },
    Normal {
        normal: [-0.682167, 0.731196, -0.00034353],
    },
    Normal {
        normal: [-0.870322, 0.492483, -0.0],
    },
    Normal {
        normal: [0.893014, -0.256345, 0.369882],
    },
    Normal {
        normal: [0.893437, 0.255997, 0.369102],
    },
    Normal {
        normal: [0.0838768, 0.995843, 0.0355066],
    },
    Normal {
        normal: [-0.629724, 0.73186, -0.260439],
    },
    Normal {
        normal: [-0.803725, 0.49337, -0.332584],
    },
    Normal {
        normal: [0.683407, -0.256729, 0.683407],
    },
    Normal {
        normal: [0.683531, 0.256067, 0.683531],
    },
    Normal {
        normal: [0.0649249, 0.995776, 0.0649249],
    },
    Normal {
        normal: [-0.481398, 0.732469, -0.481398],
    },
    Normal {
        normal: [-0.614804, 0.493997, -0.614804],
    },
    Normal {
        normal: [0.369882, -0.256345, 0.893014],
    },
    Normal {
        normal: [0.369102, 0.255997, 0.893437],
    },
    Normal {
        normal: [0.0355067, 0.995843, 0.083877],
    },
    Normal {
        normal: [-0.260439, 0.73186, -0.629724],
    },
    Normal {
        normal: [-0.332584, 0.49337, -0.803725],
    },
    Normal {
        normal: [0.0, -0.255752, 0.966742],
    },
    Normal {
        normal: [0.0, 0.255443, 0.966824],
    },
    Normal {
        normal: [0.0, 0.995754, 0.092052],
    },
    Normal {
        normal: [0.0, 0.731305, -0.68205],
    },
    Normal {
        normal: [-0.0, 0.492521, -0.870301],
    },
    Normal {
        normal: [-0.369882, -0.256345, 0.893014],
    },
    Normal {
        normal: [-0.369102, 0.255996, 0.893437],
    },
    Normal {
        normal: [-0.0355068, 0.995843, 0.0838771],
    },
    Normal {
        normal: [0.260439, 0.73186, -0.629724],
    },
    Normal {
        normal: [0.332584, 0.49337, -0.803725],
    },
    Normal {
        normal: [-0.683407, -0.256729, 0.683407],
    },
    Normal {
        normal: [-0.683531, 0.256067, 0.683531],
    },
    Normal {
        normal: [-0.0649249, 0.995776, 0.064925],
    },
    Normal {
        normal: [0.481398, 0.732469, -0.481398],
    },
    Normal {
        normal: [0.614804, 0.493997, -0.614804],
    },
    Normal {
        normal: [-0.893014, -0.256345, 0.369882],
    },
    Normal {
        normal: [-0.893437, 0.255997, 0.369102],
    },
    Normal {
        normal: [-0.0838767, 0.995843, 0.0355066],
    },
    Normal {
        normal: [0.629724, 0.73186, -0.260439],
    },
    Normal {
        normal: [0.803725, 0.49337, -0.332584],
    },
    Normal {
        normal: [0.915321, 0.402725, 0.0],
    },
    Normal {
        normal: [0.941808, 0.336151, -0.0],
    },
    Normal {
        normal: [0.97869, 0.205342, 0.0],
    },
    Normal {
        normal: [0.997804, -0.0662397, 0.0],
    },
    Normal {
        normal: [0.845438, 0.403546, 0.349835],
    },
    Normal {
        normal: [0.869996, 0.336859, 0.360047],
    },
    Normal {
        normal: [0.904193, 0.205791, 0.37428],
    },
    Normal {
        normal: [0.921879, -0.0663697, 0.381752],
    },
    Normal {
        normal: [0.646802, 0.404096, 0.646802],
    },
    Normal {
        normal: [0.665655, 0.337351, 0.665655],
    },
    Normal {
        normal: [0.691923, 0.20612, 0.691923],
    },
    Normal {
        normal: [0.705542, -0.0664796, 0.705543],
    },
    Normal {
        normal: [0.349835, 0.403546, 0.845438],
    },
    Normal {
        normal: [0.360047, 0.336859, 0.869996],
    },
    Normal {
        normal: [0.37428, 0.205791, 0.904193],
    },
    Normal {
        normal: [0.381752, -0.0663697, 0.921879],
    },
    Normal {
        normal: [-0.0, 0.402725, 0.915321],
    },
    Normal {
        normal: [0.0, 0.336151, 0.941808],
    },
    Normal {
        normal: [-0.0, 0.205342, 0.97869],
    },
    Normal {
        normal: [-0.0, -0.0662397, 0.997804],
    },
    Normal {
        normal: [-0.349835, 0.403546, 0.845438],
    },
    Normal {
        normal: [-0.360047, 0.336859, 0.869996],
    },
    Normal {
        normal: [-0.37428, 0.205791, 0.904193],
    },
    Normal {
        normal: [-0.381752, -0.0663697, 0.921879],
    },
    Normal {
        normal: [-0.646802, 0.404096, 0.646802],
    },
    Normal {
        normal: [-0.665655, 0.337351, 0.665655],
    },
    Normal {
        normal: [-0.691923, 0.20612, 0.691923],
    },
    Normal {
        normal: [-0.705543, -0.0664796, 0.705543],
    },
    Normal {
        normal: [-0.845438, 0.403546, 0.349835],
    },
    Normal {
        normal: [-0.869996, 0.336859, 0.360047],
    },
    Normal {
        normal: [-0.904193, 0.205791, 0.37428],
    },
    Normal {
        normal: [-0.921879, -0.0663697, 0.381752],
    },
    Normal {
        normal: [-0.915321, 0.402725, -0.0],
    },
    Normal {
        normal: [-0.941808, 0.336151, -0.0],
    },
    Normal {
        normal: [-0.97869, 0.205342, -0.0],
    },
    Normal {
        normal: [-0.997804, -0.0662397, -0.0],
    },
    Normal {
        normal: [-0.845438, 0.403546, -0.349835],
    },
    Normal {
        normal: [-0.869996, 0.336859, -0.360047],
    },
    Normal {
        normal: [-0.904193, 0.205791, -0.37428],
    },
    Normal {
        normal: [-0.921879, -0.0663697, -0.381752],
    },
    Normal {
        normal: [-0.646802, 0.404096, -0.646802],
    },
    Normal {
        normal: [-0.665655, 0.337351, -0.665655],
    },
    Normal {
        normal: [-0.691923, 0.20612, -0.691923],
    },
    Normal {
        normal: [-0.705542, -0.0664796, -0.705543],
    },
    Normal {
        normal: [-0.349835, 0.403546, -0.845438],
    },
    Normal {
        normal: [-0.360047, 0.336859, -0.869996],
    },
    Normal {
        normal: [-0.37428, 0.205791, -0.904193],
    },
    Normal {
        normal: [-0.381752, -0.0663697, -0.921879],
    },
    Normal {
        normal: [0.0, 0.402725, -0.915321],
    },
    Normal {
        normal: [-0.0, 0.336151, -0.941808],
    },
    Normal {
        normal: [0.0, 0.205342, -0.97869],
    },
    Normal {
        normal: [0.0, -0.0662397, -0.997804],
    },
    Normal {
        normal: [0.349835, 0.403546, -0.845438],
    },
    Normal {
        normal: [0.360047, 0.336859, -0.869996],
    },
    Normal {
        normal: [0.37428, 0.205791, -0.904193],
    },
    Normal {
        normal: [0.381752, -0.0663697, -0.921879],
    },
    Normal {
        normal: [0.646802, 0.404096, -0.646802],
    },
    Normal {
        normal: [0.665655, 0.337351, -0.665655],
    },
    Normal {
        normal: [0.691923, 0.20612, -0.691923],
    },
    Normal {
        normal: [0.705543, -0.0664796, -0.705542],
    },
    Normal {
        normal: [0.845438, 0.403546, -0.349835],
    },
    Normal {
        normal: [0.869996, 0.336859, -0.360047],
    },
    Normal {
        normal: [0.904193, 0.205791, -0.37428],
    },
    Normal {
        normal: [0.921879, -0.0663697, -0.381752],
    },
    Normal {
        normal: [0.900182, -0.435513, -0.0],
    },
    Normal {
        normal: [0.729611, -0.683863, -0.0],
    },
    Normal {
        normal: [0.693951, -0.720022, -0.0],
    },
    Normal {
        normal: [0.79395, -0.607984, 0.0],
    },
    Normal {
        normal: [0.831437, -0.43618, 0.344179],
    },
    Normal {
        normal: [0.673512, -0.684665, 0.278594],
    },
    Normal {
        normal: [0.640399, -0.720924, 0.264874],
    },
    Normal {
        normal: [0.732949, -0.608996, 0.303166],
    },
    Normal {
        normal: [0.636092, -0.436777, 0.636092],
    },
    Normal {
        normal: [0.514965, -0.685289, 0.514965],
    },
    Normal {
        normal: [0.489651, -0.721446, 0.489651],
    },
    Normal {
        normal: [0.560555, -0.609554, 0.560555],
    },
    Normal {
        normal: [0.344179, -0.43618, 0.831437],
    },
    Normal {
        normal: [0.278594, -0.684665, 0.673512],
    },
    Normal {
        normal: [0.264874, -0.720924, 0.640399],
    },
    Normal {
        normal: [0.303166, -0.608996, 0.732949],
    },
    Normal {
        normal: [0.0, -0.435513, 0.900182],
    },
    Normal {
        normal: [-0.0, -0.683863, 0.729611],
    },
    Normal {
        normal: [0.0, -0.720022, 0.693951],
    },
    Normal {
        normal: [-0.0, -0.607984, 0.79395],
    },
    Normal {
        normal: [-0.344179, -0.43618, 0.831437],
    },
    Normal {
        normal: [-0.278594, -0.684665, 0.673512],
    },
    Normal {
        normal: [-0.264874, -0.720924, 0.640399],
    },
    Normal {
        normal: [-0.303166, -0.608996, 0.732949],
    },
    Normal {
        normal: [-0.636092, -0.436777, 0.636092],
    },
    Normal {
        normal: [-0.514965, -0.685289, 0.514965],
    },
    Normal {
        normal: [-0.489651, -0.721446, 0.489651],
    },
    Normal {
        normal: [-0.560555, -0.609554, 0.560555],
    },
    Normal {
        normal: [-0.831437, -0.43618, 0.344179],
    },
    Normal {
        normal: [-0.673512, -0.684665, 0.278595],
    },
    Normal {
        normal: [-0.640399, -0.720924, 0.264874],
    },
    Normal {
        normal: [-0.732949, -0.608996, 0.303166],
    },
    Normal {
        normal: [-0.900182, -0.435513, -0.0],
    },
    Normal {
        normal: [-0.729611, -0.683863, -0.0],
    },
    Normal {
        normal: [-0.693951, -0.720022, 0.0],
    },
    Normal {
        normal: [-0.79395, -0.607983, -0.0],
    },
    Normal {
        normal: [-0.831437, -0.43618, -0.344179],
    },
    Normal {
        normal: [-0.673512, -0.684665, -0.278594],
    },
    Normal {
        normal: [-0.640399, -0.720924, -0.264874],
    },
    Normal {
        normal: [-0.732949, -0.608996, -0.303166],
    },
    Normal {
        normal: [-0.636092, -0.436777, -0.636092],
    },
    Normal {
        normal: [-0.514965, -0.685289, -0.514965],
    },
    Normal {
        normal: [-0.489651, -0.721446, -0.489651],
    },
    Normal {
        normal: [-0.560555, -0.609554, -0.560555],
    },
    Normal {
        normal: [-0.344179, -0.43618, -0.831437],
    },
    Normal {
        normal: [-0.278594, -0.684665, -0.673512],
    },
    Normal {
        normal: [-0.264874, -0.720924, -0.640399],
    },
    Normal {
        normal: [-0.303166, -0.608996, -0.732949],
    },
    Normal {
        normal: [-0.0, -0.435513, -0.900182],
    },
    Normal {
        normal: [0.0, -0.683863, -0.729611],
    },
    Normal {
        normal: [-0.0, -0.720022, -0.693951],
    },
    Normal {
        normal: [0.0, -0.607984, -0.79395],
    },
    Normal {
        normal: [0.344179, -0.43618, -0.831437],
    },
    Normal {
        normal: [0.278594, -0.684665, -0.673512],
    },
    Normal {
        normal: [0.264874, -0.720924, -0.640399],
    },
    Normal {
        normal: [0.303167, -0.608996, -0.732949],
    },
    Normal {
        normal: [0.636092, -0.436777, -0.636092],
    },
    Normal {
        normal: [0.514965, -0.685289, -0.514965],
    },
    Normal {
        normal: [0.489651, -0.721446, -0.489651],
    },
    Normal {
        normal: [0.560555, -0.609554, -0.560555],
    },
    Normal {
        normal: [0.831437, -0.43618, -0.344179],
    },
    Normal {
        normal: [0.673512, -0.684665, -0.278595],
    },
    Normal {
        normal: [0.640399, -0.720924, -0.264874],
    },
    Normal {
        normal: [0.732949, -0.608996, -0.303166],
    },
    Normal {
        normal: [0.62386, -0.781536, 0.0],
    },
    Normal {
        normal: [0.177291, -0.984159, -0.0],
    },
    Normal {
        normal: [0.0492072, -0.998789, 0.0],
    },
    Normal {
        normal: [0.0, -1.0, -0.0],
    },
    Normal {
        normal: [0.576229, -0.781801, 0.238217],
    },
    Normal {
        normal: [0.163629, -0.984208, 0.0675273],
    },
    Normal {
        normal: [0.0454217, -0.998792, 0.0187357],
    },
    Normal {
        normal: [0.440416, -0.782348, 0.440416],
    },
    Normal {
        normal: [0.124903, -0.984276, 0.124903],
    },
    Normal {
        normal: [0.0346621, -0.998798, 0.0346621],
    },
    Normal {
        normal: [0.238217, -0.781801, 0.576229],
    },
    Normal {
        normal: [0.0675273, -0.984208, 0.163629],
    },
    Normal {
        normal: [0.0187357, -0.998792, 0.0454217],
    },
    Normal {
        normal: [-0.0, -0.781536, 0.62386],
    },
    Normal {
        normal: [0.0, -0.984159, 0.177291],
    },
    Normal {
        normal: [-0.0, -0.998789, 0.0492072],
    },
    Normal {
        normal: [-0.238216, -0.781801, 0.576229],
    },
    Normal {
        normal: [-0.0675273, -0.984208, 0.163629],
    },
    Normal {
        normal: [-0.0187357, -0.998792, 0.0454217],
    },
    Normal {
        normal: [-0.440416, -0.782348, 0.440416],
    },
    Normal {
        normal: [-0.124903, -0.984276, 0.124903],
    },
    Normal {
        normal: [-0.0346621, -0.998798, 0.0346621],
    },
    Normal {
        normal: [-0.576229, -0.781801, 0.238217],
    },
    Normal {
        normal: [-0.163629, -0.984208, 0.0675273],
    },
    Normal {
        normal: [-0.0454217, -0.998792, 0.0187357],
    },
    Normal {
        normal: [-0.62386, -0.781536, -0.0],
    },
    Normal {
        normal: [-0.177291, -0.984159, 0.0],
    },
    Normal {
        normal: [-0.0492072, -0.998789, -0.0],
    },
    Normal {
        normal: [-0.576229, -0.781801, -0.238217],
    },
    Normal {
        normal: [-0.163629, -0.984208, -0.0675273],
    },
    Normal {
        normal: [-0.0454217, -0.998792, -0.0187357],
    },
    Normal {
        normal: [-0.440416, -0.782348, -0.440416],
    },
    Normal {
        normal: [-0.124903, -0.984276, -0.124903],
    },
    Normal {
        normal: [-0.0346621, -0.998798, -0.0346621],
    },
    Normal {
        normal: [-0.238217, -0.781801, -0.576229],
    },
    Normal {
        normal: [-0.0675273, -0.984208, -0.163629],
    },
    Normal {
        normal: [-0.0187357, -0.998792, -0.0454217],
    },
    Normal {
        normal: [0.0, -0.781536, -0.62386],
    },
    Normal {
        normal: [-0.0, -0.984159, -0.177291],
    },
    Normal {
        normal: [0.0, -0.998789, -0.0492072],
    },
    Normal {
        normal: [0.238217, -0.781801, -0.576229],
    },
    Normal {
        normal: [0.0675273, -0.984208, -0.163629],
    },
    Normal {
        normal: [0.0187357, -0.998792, -0.0454217],
    },
    Normal {
        normal: [0.440416, -0.782348, -0.440416],
    },
    Normal {
        normal: [0.124903, -0.984276, -0.124903],
    },
    Normal {
        normal: [0.0346621, -0.998798, -0.0346621],
    },
    Normal {
        normal: [0.576229, -0.781801, -0.238217],
    },
    Normal {
        normal: [0.163629, -0.984208, -0.0675273],
    },
    Normal {
        normal: [0.0454217, -0.998792, -0.0187357],
    },
    Normal {
        normal: [0.00778619, -0.99997, -0.000215809],
    },
    Normal {
        normal: [0.0391385, -0.999233, -0.000988567],
    },
    Normal {
        normal: [0.179511, -0.983746, -0.00436856],
    },
    Normal {
        normal: [0.6123, -0.790556, -0.0104598],
    },
    Normal {
        normal: [0.986152, -0.165707, -0.00666949],
    },
    Normal {
        normal: [0.00703893, -0.812495, 0.582926],
    },
    Normal {
        normal: [0.0361273, -0.837257, 0.545614],
    },
    Normal {
        normal: [0.161845, -0.810421, 0.563048],
    },
    Normal {
        normal: [0.482365, -0.595148, 0.642746],
    },
    Normal {
        normal: [0.73872, -0.114593, 0.664199],
    },
    Normal {
        normal: [-0.00190867, 0.162121, 0.986769],
    },
    Normal {
        normal: [0.0027616, 0.0171073, 0.99985],
    },
    Normal {
        normal: [0.0105326, 0.0733989, 0.997247],
    },
    Normal {
        normal: [-0.0660406, 0.130069, 0.989303],
    },
    Normal {
        normal: [-0.0944272, 0.0165946, 0.995393],
    },
    Normal {
        normal: [-0.009203, 0.871509, 0.490293],
    },
    Normal {
        normal: [-0.0486064, 0.840609, 0.539457],
    },
    Normal {
        normal: [-0.223298, 0.802881, 0.552739],
    },
    Normal {
        normal: [-0.596365, 0.559971, 0.575135],
    },
    Normal {
        normal: [-0.803337, 0.0682361, 0.591602],
    },
    Normal {
        normal: [-0.0105609, 0.999944, 0.000103364],
    },
    Normal {
        normal: [-0.0587986, 0.99827, 0.000709759],
    },
    Normal {
        normal: [-0.28071, 0.959787, 0.00326876],
    },
    Normal {
        normal: [-0.749723, 0.661738, 0.0042684],
    },
    Normal {
        normal: [-0.997351, 0.0727144, 0.00205923],
    },
    Normal {
        normal: [-0.00879197, 0.871493, -0.49033],
    },
    Normal {
        normal: [-0.0464937, 0.841178, -0.538756],
    },
    Normal {
        normal: [-0.217909, 0.806807, -0.549161],
    },
    Normal {
        normal: [-0.597291, 0.560026, -0.574121],
    },
    Normal {
        normal: [-0.804, 0.0629127, -0.591291],
    },
    Normal {
        normal: [-0.00180555, 0.161691, -0.98684],
    },
    Normal {
        normal: [0.00203087, 0.014555, -0.999892],
    },
    Normal {
        normal: [0.00921499, 0.0600698, -0.998152],
    },
    Normal {
        normal: [-0.0593333, 0.113865, -0.991723],
    },
    Normal {
        normal: [-0.0868992, 0.0122903, -0.996141],
    },
    Normal {
        normal: [0.00641779, -0.812379, -0.583094],
    },
    Normal {
        normal: [0.0337833, -0.837512, -0.545373],
    },
    Normal {
        normal: [0.157112, -0.811947, -0.56219],
    },
    Normal {
        normal: [0.484407, -0.589365, -0.646528],
    },
    Normal {
        normal: [0.73887, -0.10132, -0.666187],
    },
    Normal {
        normal: [0.946512, 0.32265, -0.0033571],
    },
    Normal {
        normal: [0.82583, 0.56387, -0.00745213],
    },
    Normal {
        normal: [0.650011, 0.759893, -0.00693681],
    },
    Normal {
        normal: [0.532429, 0.846458, -0.00524544],
    },
    Normal {
        normal: [0.725608, 0.259351, 0.637362],
    },
    Normal {
        normal: [0.645945, 0.461988, 0.607719],
    },
    Normal {
        normal: [0.531614, 0.63666, 0.558615],
    },
    Normal {
        normal: [0.424964, 0.681717, 0.59554],
    },
    Normal {
        normal: [-0.0495616, -0.019755, 0.998576],
    },
    Normal {
        normal: [-0.0378162, -0.0356243, 0.99865],
    },
    Normal {
        normal: [-0.0379139, -0.0365122, 0.998614],
    },
    Normal {
        normal: [-0.168854, -0.297946, 0.93953],
    },
    Normal {
        normal: [-0.742342, -0.299166, 0.599523],
    },
    Normal {
        normal: [-0.619602, -0.529406, 0.579503],
    },
    Normal {
        normal: [-0.483708, -0.685761, 0.543837],
    },
    Normal {
        normal: [-0.445293, -0.794355, 0.413176],
    },
    Normal {
        normal: [-0.926513, -0.376257, 0.00199587],
    },
    Normal {
        normal: [-0.75392, -0.656952, 0.00431723],
    },
    Normal {
        normal: [-0.566224, -0.824244, 0.00346105],
    },
    Normal {
        normal: [-0.481804, -0.876277, 0.00185047],
    },
    Normal {
        normal: [-0.744675, -0.294424, -0.598977],
    },
    Normal {
        normal: [-0.621949, -0.528114, -0.578165],
    },
    Normal {
        normal: [-0.481171, -0.68834, -0.542828],
    },
    Normal {
        normal: [-0.438055, -0.797035, -0.415744],
    },
    Normal {
        normal: [-0.0443368, -0.0170558, -0.998871],
    },
    Normal {
        normal: [-0.0261761, -0.0281665, -0.99926],
    },
    Normal {
        normal: [-0.0252939, -0.0283323, -0.999278],
    },
    Normal {
        normal: [-0.157482, -0.289392, -0.944167],
    },
    Normal {
        normal: [0.728244, 0.25241, -0.637142],
    },
    Normal {
        normal: [0.647055, 0.459725, -0.608254],
    },
    Normal {
        normal: [0.522994, 0.640657, -0.562171],
    },
    Normal {
        normal: [0.409978, 0.682857, -0.604669],
    },
    Normal {
        normal: [-0.230787, 0.972982, -0.00652338],
    },
    Normal {
        normal: [-0.548936, 0.835863, -0.00151111],
    },
    Normal {
        normal: [-0.875671, 0.482807, 0.00989278],
    },
    Normal {
        normal: [-0.877554, 0.479097, 0.0190923],
    },
    Normal {
        normal: [-0.69619, 0.717439, 0.024497],
    },
    Normal {
        normal: [-0.152878, 0.687211, 0.71019],
    },
    Normal {
        normal: [-0.316721, 0.63775, 0.702113],
    },
    Normal {
        normal: [-0.601067, 0.471452, 0.64533],
    },
    Normal {
        normal: [-0.635889, 0.44609, 0.6298],
    },
    Normal {
        normal: [-0.435746, 0.601008, 0.670011],
    },
    Normal {
        normal: [0.111112, -0.0850694, 0.99016],
    },
    Normal {
        normal: [0.22331, 0.00654036, 0.974726],
    },
    Normal {
        normal: [0.190097, 0.154964, 0.969458],
    },
    Normal {
        normal: [0.00527077, 0.189482, 0.98187],
    },
    Normal {
        normal: [-0.0117518, 0.246688, 0.969024],
    },
    Normal {
        normal: [0.343906, -0.722796, 0.599412],
    },
    Normal {
        normal: [0.572489, -0.567656, 0.591627],
    },
    Normal {
        normal: [0.787436, -0.256459, 0.560512],
    },
    Normal {
        normal: [0.647097, -0.306374, 0.698141],
    },
    Normal {
        normal: [0.427528, -0.499343, 0.753576],
    },
    Normal {
        normal: [0.410926, -0.911668, 0.00128446],
    },
    Normal {
        normal: [0.67152, -0.740986, -0.000899122],
    },
    Normal {
        normal: [0.922026, -0.38706, -0.00725269],
    },
    Normal {
        normal: [0.84691, -0.531556, -0.0138542],
    },
    Normal {
        normal: [0.535925, -0.8442, -0.0105045],
    },
    Normal {
        normal: [0.341188, -0.722822, -0.600931],
    },
    Normal {
        normal: [0.578664, -0.561139, -0.591838],
    },
    Normal {
        normal: [0.784869, -0.25102, -0.566542],
    },
    Normal {
        normal: [0.642681, -0.302257, -0.70399],
    },
    Normal {
        normal: [0.418589, -0.500042, -0.758117],
    },
    Normal {
        normal: [0.115806, -0.0791394, -0.990114],
    },
    Normal {
        normal: [0.232811, 0.0125652, -0.972441],
    },
    Normal {
        normal: [0.206662, 0.153601, -0.96628],
    },
    Normal {
        normal: [0.0244996, 0.161443, -0.986578],
    },
    Normal {
        normal: [0.00338193, 0.211115, -0.977455],
    },
    Normal {
        normal: [-0.134912, 0.687491, -0.713551],
    },
    Normal {
        normal: [-0.31954, 0.633073, -0.705062],
    },
    Normal {
        normal: [-0.603902, 0.461442, -0.649903],
    },
    Normal {
        normal: [-0.631816, 0.437169, -0.640072],
    },
    Normal {
        normal: [-0.424306, 0.612706, -0.66675],
    },
    Normal {
        normal: [-0.4258, 0.904753, 0.0108049],
    },
    Normal {
        normal: [0.0220472, 0.999756, 0.00162273],
    },
    Normal {
        normal: [0.999599, 0.0258705, 0.0115556],
    },
    Normal {
        normal: [0.709585, -0.704553, 0.00967183],
    },
    Normal {
        normal: [-0.259858, 0.791936, 0.552549],
    },
    Normal {
        normal: [0.00953916, 0.99972, -0.0216718],
    },
    Normal {
        normal: [0.410156, 0.332912, -0.849083],
    },
    Normal {
        normal: [0.541523, -0.54862, -0.637],
    },
    Normal {
        normal: [0.0463104, 0.455224, 0.889172],
    },
    Normal {
        normal: [-0.0106883, 0.988794, 0.148901],
    },
    Normal {
        normal: [-0.0443756, 0.682947, -0.729118],
    },
    Normal {
        normal: [0.122825, 0.00923214, -0.992385],
    },
    Normal {
        normal: [0.481839, -0.180439, 0.85748],
    },
    Normal {
        normal: [0.455272, 0.736752, 0.499925],
    },
    Normal {
        normal: [-0.220542, 0.907193, -0.358276],
    },
    Normal {
        normal: [-0.23592, 0.657249, -0.715797],
    },
    Normal {
        normal: [0.728092, -0.685302, -0.0155853],
    },
    Normal {
        normal: [0.888739, 0.45811, -0.0166791],
    },
    Normal {
        normal: [-0.260097, 0.965582, 0.000800195],
    },
    Normal {
        normal: [-0.371612, 0.928378, -0.00441745],
    },
    Normal {
        normal: [0.480166, -0.17836, -0.858853],
    },
    Normal {
        normal: [0.488103, 0.716801, -0.497947],
    },
    Normal {
        normal: [-0.222004, 0.905399, 0.361893],
    },
    Normal {
        normal: [-0.235405, 0.66318, 0.710477],
    },
    Normal {
        normal: [0.0587203, 0.437704, -0.8972],
    },
    Normal {
        normal: [0.00132612, 0.986459, -0.164003],
    },
    Normal {
        normal: [-0.0441901, 0.681677, 0.730317],
    },
    Normal {
        normal: [0.138801, -0.0341896, 0.98973],
    },
    Normal {
        normal: [-0.25889, 0.797206, -0.54538],
    },
    Normal {
        normal: [0.0122703, 0.999739, 0.0192865],
    },
    Normal {
        normal: [0.39863, 0.35489, 0.845663],
    },
    Normal {
        normal: [0.537564, -0.5814, 0.610737],
    },
    Normal {
        normal: [-0.0, 1.0, 0.0],
    },
    Normal {
        normal: [0.82454, 0.565804, 0.0],
    },
    Normal {
        normal: [0.917701, -0.397272, 0.0],
    },
    Normal {
        normal: [0.935269, -0.353939, 0.000112842],
    },
    Normal {
        normal: [0.780712, 0.624891, 0.0],
    },
    Normal {
        normal: [0.762641, 0.565035, 0.314825],
    },
    Normal {
        normal: [0.847982, -0.397998, 0.350034],
    },
    Normal {
        normal: [0.864141, -0.355261, 0.356441],
    },
    Normal {
        normal: [0.720991, 0.625625, 0.297933],
    },
    Normal {
        normal: [0.583357, 0.565165, 0.583338],
    },
    Normal {
        normal: [0.648485, -0.398726, 0.648448],
    },
    Normal {
        normal: [0.660872, -0.355894, 0.660748],
    },
    Normal {
        normal: [0.551862, 0.62529, 0.55178],
    },
    Normal {
        normal: [0.314824, 0.565051, 0.762629],
    },
    Normal {
        normal: [0.350045, -0.397976, 0.847988],
    },
    Normal {
        normal: [0.356474, -0.3552, 0.864153],
    },
    Normal {
        normal: [0.297983, 0.625515, 0.721067],
    },
    Normal {
        normal: [-0.0, 0.565804, 0.82454],
    },
    Normal {
        normal: [-0.0, -0.397272, 0.917701],
    },
    Normal {
        normal: [-0.000112839, -0.353939, 0.935269],
    },
    Normal {
        normal: [-0.0, 0.624891, 0.780712],
    },
    Normal {
        normal: [-0.314825, 0.565035, 0.762641],
    },
    Normal {
        normal: [-0.350034, -0.397998, 0.847982],
    },
    Normal {
        normal: [-0.356441, -0.355261, 0.864141],
    },
    Normal {
        normal: [-0.297933, 0.625625, 0.720991],
    },
    Normal {
        normal: [-0.583338, 0.565165, 0.583357],
    },
    Normal {
        normal: [-0.648448, -0.398726, 0.648485],
    },
    Normal {
        normal: [-0.660748, -0.355894, 0.660872],
    },
    Normal {
        normal: [-0.55178, 0.62529, 0.551862],
    },
    Normal {
        normal: [-0.762629, 0.565051, 0.314824],
    },
    Normal {
        normal: [-0.847988, -0.397976, 0.350045],
    },
    Normal {
        normal: [-0.864153, -0.3552, 0.356474],
    },
    Normal {
        normal: [-0.721067, 0.625515, 0.297983],
    },
    Normal {
        normal: [-0.82454, 0.565804, -0.0],
    },
    Normal {
        normal: [-0.917701, -0.397272, -0.0],
    },
    Normal {
        normal: [-0.935269, -0.353939, -0.000112839],
    },
    Normal {
        normal: [-0.780712, 0.624891, -0.0],
    },
    Normal {
        normal: [-0.76264, 0.565035, -0.314825],
    },
    Normal {
        normal: [-0.847982, -0.397998, -0.350034],
    },
    Normal {
        normal: [-0.864141, -0.355261, -0.356441],
    },
    Normal {
        normal: [-0.720991, 0.625625, -0.297933],
    },
    Normal {
        normal: [-0.583357, 0.565165, -0.583338],
    },
    Normal {
        normal: [-0.648485, -0.398726, -0.648448],
    },
    Normal {
        normal: [-0.660872, -0.355894, -0.660748],
    },
    Normal {
        normal: [-0.551862, 0.62529, -0.55178],
    },
    Normal {
        normal: [-0.314824, 0.565051, -0.762629],
    },
    Normal {
        normal: [-0.350045, -0.397976, -0.847988],
    },
    Normal {
        normal: [-0.356474, -0.3552, -0.864153],
    },
    Normal {
        normal: [-0.297983, 0.625515, -0.721067],
    },
    Normal {
        normal: [0.0, 0.565804, -0.82454],
    },
    Normal {
        normal: [0.0, -0.397272, -0.917701],
    },
    Normal {
        normal: [0.000112839, -0.353939, -0.935269],
    },
    Normal {
        normal: [0.0, 0.624891, -0.780712],
    },
    Normal {
        normal: [0.314825, 0.565035, -0.762641],
    },
    Normal {
        normal: [0.350034, -0.397998, -0.847982],
    },
    Normal {
        normal: [0.356441, -0.355261, -0.864141],
    },
    Normal {
        normal: [0.297933, 0.625625, -0.720991],
    },
    Normal {
        normal: [0.583338, 0.565165, -0.583357],
    },
    Normal {
        normal: [0.648448, -0.398726, -0.648485],
    },
    Normal {
        normal: [0.660748, -0.355894, -0.660872],
    },
    Normal {
        normal: [0.55178, 0.62529, -0.551862],
    },
    Normal {
        normal: [0.762629, 0.565051, -0.314824],
    },
    Normal {
        normal: [0.847988, -0.397976, -0.350045],
    },
    Normal {
        normal: [0.864153, -0.3552, -0.356474],
    },
    Normal {
        normal: [0.721067, 0.625515, -0.297983],
    },
    Normal {
        normal: [0.236584, 0.971611, 0.0],
    },
    Normal {
        normal: [0.173084, 0.984907, -0.0],
    },
    Normal {
        normal: [0.379703, 0.925108, 0.0],
    },
    Normal {
        normal: [0.526673, 0.850068, 0.0],
    },
    Normal {
        normal: [0.217978, 0.971775, 0.0902162],
    },
    Normal {
        normal: [0.15959, 0.984977, 0.0659615],
    },
    Normal {
        normal: [0.350498, 0.925312, 0.14474],
    },
    Normal {
        normal: [0.48559, 0.850653, 0.201474],
    },
    Normal {
        normal: [0.166631, 0.971838, 0.166631],
    },
    Normal {
        normal: [0.121908, 0.985026, 0.121908],
    },
    Normal {
        normal: [0.267668, 0.925585, 0.267668],
    },
    Normal {
        normal: [0.371315, 0.851029, 0.371315],
    },
    Normal {
        normal: [0.0902162, 0.971775, 0.217978],
    },
    Normal {
        normal: [0.0659615, 0.984977, 0.15959],
    },
    Normal {
        normal: [0.14474, 0.925312, 0.350498],
    },
    Normal {
        normal: [0.201474, 0.850653, 0.48559],
    },
    Normal {
        normal: [-0.0, 0.971611, 0.236584],
    },
    Normal {
        normal: [0.0, 0.984907, 0.173084],
    },
    Normal {
        normal: [0.0, 0.925108, 0.379703],
    },
    Normal {
        normal: [0.0, 0.850068, 0.526673],
    },
    Normal {
        normal: [-0.0902162, 0.971775, 0.217978],
    },
    Normal {
        normal: [-0.0659615, 0.984977, 0.15959],
    },
    Normal {
        normal: [-0.14474, 0.925312, 0.350498],
    },
    Normal {
        normal: [-0.201474, 0.850653, 0.48559],
    },
    Normal {
        normal: [-0.166631, 0.971838, 0.166631],
    },
    Normal {
        normal: [-0.121908, 0.985026, 0.121908],
    },
    Normal {
        normal: [-0.267668, 0.925585, 0.267668],
    },
    Normal {
        normal: [-0.371315, 0.851029, 0.371315],
    },
    Normal {
        normal: [-0.217978, 0.971775, 0.0902162],
    },
    Normal {
        normal: [-0.15959, 0.984977, 0.0659615],
    },
    Normal {
        normal: [-0.350498, 0.925312, 0.14474],
    },
    Normal {
        normal: [-0.48559, 0.850653, 0.201474],
    },
    Normal {
        normal: [-0.236583, 0.971611, -0.0],
    },
    Normal {
        normal: [-0.173084, 0.984907, 0.0],
    },
    Normal {
        normal: [-0.379703, 0.925108, -0.0],
    },
    Normal {
        normal: [-0.526673, 0.850068, 0.0],
    },
    Normal {
        normal: [-0.217978, 0.971775, -0.0902162],
    },
    Normal {
        normal: [-0.15959, 0.984977, -0.0659615],
    },
    Normal {
        normal: [-0.350498, 0.925312, -0.14474],
    },
    Normal {
        normal: [-0.48559, 0.850653, -0.201474],
    },
    Normal {
        normal: [-0.166631, 0.971838, -0.166631],
    },
    Normal {
        normal: [-0.121908, 0.985026, -0.121908],
    },
    Normal {
        normal: [-0.267668, 0.925585, -0.267668],
    },
    Normal {
        normal: [-0.371315, 0.851029, -0.371315],
    },
    Normal {
        normal: [-0.0902162, 0.971775, -0.217978],
    },
    Normal {
        normal: [-0.0659615, 0.984977, -0.15959],
    },
    Normal {
        normal: [-0.14474, 0.925312, -0.350498],
    },
    Normal {
        normal: [-0.201474, 0.850653, -0.485589],
    },
    Normal {
        normal: [0.0, 0.971611, -0.236584],
    },
    Normal {
        normal: [-0.0, 0.984907, -0.173084],
    },
    Normal {
        normal: [-0.0, 0.925108, -0.379703],
    },
    Normal {
        normal: [-0.0, 0.850068, -0.526673],
    },
    Normal {
        normal: [0.0902162, 0.971775, -0.217978],
    },
    Normal {
        normal: [0.0659615, 0.984977, -0.15959],
    },
    Normal {
        normal: [0.14474, 0.925312, -0.350498],
    },
    Normal {
        normal: [0.201474, 0.850653, -0.48559],
    },
    Normal {
        normal: [0.166631, 0.971838, -0.166631],
    },
    Normal {
        normal: [0.121908, 0.985026, -0.121908],
    },
    Normal {
        normal: [0.267668, 0.925585, -0.267668],
    },
    Normal {
        normal: [0.371315, 0.851029, -0.371315],
    },
    Normal {
        normal: [0.217978, 0.971775, -0.0902162],
    },
    Normal {
        normal: [0.15959, 0.984977, -0.0659615],
    },
    Normal {
        normal: [0.350498, 0.925312, -0.14474],
    },
    Normal {
        normal: [0.48559, 0.850653, -0.201474],
    },
];

pub const INDICES: [u16; 3072] = [
    7, 6, 1, 1, 2, 7, 8, 7, 2, 2, 3, 8, 9, 8, 3, 3, 4, 9, 10, 9, 4, 4, 5, 10, 12, 11, 6, 6, 7, 12,
    13, 12, 7, 7, 8, 13, 14, 13, 8, 8, 9, 14, 15, 14, 9, 9, 10, 15, 17, 16, 11, 11, 12, 17, 18, 17,
    12, 12, 13, 18, 19, 18, 13, 13, 14, 19, 20, 19, 14, 14, 15, 20, 22, 21, 16, 16, 17, 22, 23, 22,
    17, 17, 18, 23, 24, 23, 18, 18, 19, 24, 25, 24, 19, 19, 20, 25, 27, 26, 21, 21, 22, 27, 28, 27,
    22, 22, 23, 28, 29, 28, 23, 23, 24, 29, 30, 29, 24, 24, 25, 30, 32, 31, 26, 26, 27, 32, 33, 32,
    27, 27, 28, 33, 34, 33, 28, 28, 29, 34, 35, 34, 29, 29, 30, 35, 37, 36, 31, 31, 32, 37, 38, 37,
    32, 32, 33, 38, 39, 38, 33, 33, 34, 39, 40, 39, 34, 34, 35, 40, 42, 41, 36, 36, 37, 42, 43, 42,
    37, 37, 38, 43, 44, 43, 38, 38, 39, 44, 45, 44, 39, 39, 40, 45, 47, 46, 41, 41, 42, 47, 48, 47,
    42, 42, 43, 48, 49, 48, 43, 43, 44, 49, 50, 49, 44, 44, 45, 50, 52, 51, 46, 46, 47, 52, 53, 52,
    47, 47, 48, 53, 54, 53, 48, 48, 49, 54, 55, 54, 49, 49, 50, 55, 57, 56, 51, 51, 52, 57, 58, 57,
    52, 52, 53, 58, 59, 58, 53, 53, 54, 59, 60, 59, 54, 54, 55, 60, 62, 61, 56, 56, 57, 62, 63, 62,
    57, 57, 58, 63, 64, 63, 58, 58, 59, 64, 65, 64, 59, 59, 60, 65, 67, 66, 61, 61, 62, 67, 68, 67,
    62, 62, 63, 68, 69, 68, 63, 63, 64, 69, 70, 69, 64, 64, 65, 70, 72, 71, 66, 66, 67, 72, 73, 72,
    67, 67, 68, 73, 74, 73, 68, 68, 69, 74, 75, 74, 69, 69, 70, 75, 77, 76, 71, 71, 72, 77, 78, 77,
    72, 72, 73, 78, 79, 78, 73, 73, 74, 79, 80, 79, 74, 74, 75, 80, 2, 1, 76, 76, 77, 2, 3, 2, 77,
    77, 78, 3, 4, 3, 78, 78, 79, 4, 5, 4, 79, 79, 80, 5, 85, 10, 5, 5, 81, 85, 86, 85, 81, 81, 82,
    86, 87, 86, 82, 82, 83, 87, 88, 87, 83, 83, 84, 88, 89, 15, 10, 10, 85, 89, 90, 89, 85, 85, 86,
    90, 91, 90, 86, 86, 87, 91, 92, 91, 87, 87, 88, 92, 93, 20, 15, 15, 89, 93, 94, 93, 89, 89, 90,
    94, 95, 94, 90, 90, 91, 95, 96, 95, 91, 91, 92, 96, 97, 25, 20, 20, 93, 97, 98, 97, 93, 93, 94,
    98, 99, 98, 94, 94, 95, 99, 100, 99, 95, 95, 96, 100, 101, 30, 25, 25, 97, 101, 102, 101, 97,
    97, 98, 102, 103, 102, 98, 98, 99, 103, 104, 103, 99, 99, 100, 104, 105, 35, 30, 30, 101, 105,
    106, 105, 101, 101, 102, 106, 107, 106, 102, 102, 103, 107, 108, 107, 103, 103, 104, 108, 109,
    40, 35, 35, 105, 109, 110, 109, 105, 105, 106, 110, 111, 110, 106, 106, 107, 111, 112, 111,
    107, 107, 108, 112, 113, 45, 40, 40, 109, 113, 114, 113, 109, 109, 110, 114, 115, 114, 110,
    110, 111, 115, 116, 115, 111, 111, 112, 116, 117, 50, 45, 45, 113, 117, 118, 117, 113, 113,
    114, 118, 119, 118, 114, 114, 115, 119, 120, 119, 115, 115, 116, 120, 121, 55, 50, 50, 117,
    121, 122, 121, 117, 117, 118, 122, 123, 122, 118, 118, 119, 123, 124, 123, 119, 119, 120, 124,
    125, 60, 55, 55, 121, 125, 126, 125, 121, 121, 122, 126, 127, 126, 122, 122, 123, 127, 128,
    127, 123, 123, 124, 128, 129, 65, 60, 60, 125, 129, 130, 129, 125, 125, 126, 130, 131, 130,
    126, 126, 127, 131, 132, 131, 127, 127, 128, 132, 133, 70, 65, 65, 129, 133, 134, 133, 129,
    129, 130, 134, 135, 134, 130, 130, 131, 135, 136, 135, 131, 131, 132, 136, 137, 75, 70, 70,
    133, 137, 138, 137, 133, 133, 134, 138, 139, 138, 134, 134, 135, 139, 140, 139, 135, 135, 136,
    140, 141, 80, 75, 75, 137, 141, 142, 141, 137, 137, 138, 142, 143, 142, 138, 138, 139, 143,
    144, 143, 139, 139, 140, 144, 81, 5, 80, 80, 141, 81, 82, 81, 141, 141, 142, 82, 83, 82, 142,
    142, 143, 83, 84, 83, 143, 143, 144, 84, 149, 88, 84, 84, 145, 149, 150, 149, 145, 145, 146,
    150, 151, 150, 146, 146, 147, 151, 152, 151, 147, 147, 148, 152, 153, 92, 88, 88, 149, 153,
    154, 153, 149, 149, 150, 154, 155, 154, 150, 150, 151, 155, 156, 155, 151, 151, 152, 156, 157,
    96, 92, 92, 153, 157, 158, 157, 153, 153, 154, 158, 159, 158, 154, 154, 155, 159, 160, 159,
    155, 155, 156, 160, 161, 100, 96, 96, 157, 161, 162, 161, 157, 157, 158, 162, 163, 162, 158,
    158, 159, 163, 164, 163, 159, 159, 160, 164, 165, 104, 100, 100, 161, 165, 166, 165, 161, 161,
    162, 166, 167, 166, 162, 162, 163, 167, 168, 167, 163, 163, 164, 168, 169, 108, 104, 104, 165,
    169, 170, 169, 165, 165, 166, 170, 171, 170, 166, 166, 167, 171, 172, 171, 167, 167, 168, 172,
    173, 112, 108, 108, 169, 173, 174, 173, 169, 169, 170, 174, 175, 174, 170, 170, 171, 175, 176,
    175, 171, 171, 172, 176, 177, 116, 112, 112, 173, 177, 178, 177, 173, 173, 174, 178, 179, 178,
    174, 174, 175, 179, 180, 179, 175, 175, 176, 180, 181, 120, 116, 116, 177, 181, 182, 181, 177,
    177, 178, 182, 183, 182, 178, 178, 179, 183, 184, 183, 179, 179, 180, 184, 185, 124, 120, 120,
    181, 185, 186, 185, 181, 181, 182, 186, 187, 186, 182, 182, 183, 187, 188, 187, 183, 183, 184,
    188, 189, 128, 124, 124, 185, 189, 190, 189, 185, 185, 186, 190, 191, 190, 186, 186, 187, 191,
    192, 191, 187, 187, 188, 192, 193, 132, 128, 128, 189, 193, 194, 193, 189, 189, 190, 194, 195,
    194, 190, 190, 191, 195, 196, 195, 191, 191, 192, 196, 197, 136, 132, 132, 193, 197, 198, 197,
    193, 193, 194, 198, 199, 198, 194, 194, 195, 199, 200, 199, 195, 195, 196, 200, 201, 140, 136,
    136, 197, 201, 202, 201, 197, 197, 198, 202, 203, 202, 198, 198, 199, 203, 204, 203, 199, 199,
    200, 204, 205, 144, 140, 140, 201, 205, 206, 205, 201, 201, 202, 206, 207, 206, 202, 202, 203,
    207, 208, 207, 203, 203, 204, 208, 145, 84, 144, 144, 205, 145, 146, 145, 205, 205, 206, 146,
    147, 146, 206, 206, 207, 147, 148, 147, 207, 207, 208, 148, 213, 152, 148, 148, 209, 213, 214,
    213, 209, 209, 210, 214, 215, 214, 210, 210, 211, 215, 212, 215, 211, 211, 212, 212, 216, 156,
    152, 152, 213, 216, 217, 216, 213, 213, 214, 217, 218, 217, 214, 214, 215, 218, 212, 218, 215,
    215, 212, 212, 219, 160, 156, 156, 216, 219, 220, 219, 216, 216, 217, 220, 221, 220, 217, 217,
    218, 221, 212, 221, 218, 218, 212, 212, 222, 164, 160, 160, 219, 222, 223, 222, 219, 219, 220,
    223, 224, 223, 220, 220, 221, 224, 212, 224, 221, 221, 212, 212, 225, 168, 164, 164, 222, 225,
    226, 225, 222, 222, 223, 226, 227, 226, 223, 223, 224, 227, 212, 227, 224, 224, 212, 212, 228,
    172, 168, 168, 225, 228, 229, 228, 225, 225, 226, 229, 230, 229, 226, 226, 227, 230, 212, 230,
    227, 227, 212, 212, 231, 176, 172, 172, 228, 231, 232, 231, 228, 228, 229, 232, 233, 232, 229,
    229, 230, 233, 212, 233, 230, 230, 212, 212, 234, 180, 176, 176, 231, 234, 235, 234, 231, 231,
    232, 235, 236, 235, 232, 232, 233, 236, 212, 236, 233, 233, 212, 212, 237, 184, 180, 180, 234,
    237, 238, 237, 234, 234, 235, 238, 239, 238, 235, 235, 236, 239, 212, 239, 236, 236, 212, 212,
    240, 188, 184, 184, 237, 240, 241, 240, 237, 237, 238, 241, 242, 241, 238, 238, 239, 242, 212,
    242, 239, 239, 212, 212, 243, 192, 188, 188, 240, 243, 244, 243, 240, 240, 241, 244, 245, 244,
    241, 241, 242, 245, 212, 245, 242, 242, 212, 212, 246, 196, 192, 192, 243, 246, 247, 246, 243,
    243, 244, 247, 248, 247, 244, 244, 245, 248, 212, 248, 245, 245, 212, 212, 249, 200, 196, 196,
    246, 249, 250, 249, 246, 246, 247, 250, 251, 250, 247, 247, 248, 251, 212, 251, 248, 248, 212,
    212, 252, 204, 200, 200, 249, 252, 253, 252, 249, 249, 250, 253, 254, 253, 250, 250, 251, 254,
    212, 254, 251, 251, 212, 212, 255, 208, 204, 204, 252, 255, 256, 255, 252, 252, 253, 256, 257,
    256, 253, 253, 254, 257, 212, 257, 254, 254, 212, 212, 209, 148, 208, 208, 255, 209, 210, 209,
    255, 255, 256, 210, 211, 210, 256, 256, 257, 211, 212, 211, 257, 257, 212, 212, 264, 263, 258,
    258, 259, 264, 265, 264, 259, 259, 260, 265, 266, 265, 260, 260, 261, 266, 267, 266, 261, 261,
    262, 267, 269, 268, 263, 263, 264, 269, 270, 269, 264, 264, 265, 270, 271, 270, 265, 265, 266,
    271, 272, 271, 266, 266, 267, 272, 274, 273, 268, 268, 269, 274, 275, 274, 269, 269, 270, 275,
    276, 275, 270, 270, 271, 276, 277, 276, 271, 271, 272, 277, 279, 278, 273, 273, 274, 279, 280,
    279, 274, 274, 275, 280, 281, 280, 275, 275, 276, 281, 282, 281, 276, 276, 277, 282, 284, 283,
    278, 278, 279, 284, 285, 284, 279, 279, 280, 285, 286, 285, 280, 280, 281, 286, 287, 286, 281,
    281, 282, 287, 289, 288, 283, 283, 284, 289, 290, 289, 284, 284, 285, 290, 291, 290, 285, 285,
    286, 291, 292, 291, 286, 286, 287, 292, 294, 293, 288, 288, 289, 294, 295, 294, 289, 289, 290,
    295, 296, 295, 290, 290, 291, 296, 297, 296, 291, 291, 292, 297, 259, 258, 293, 293, 294, 259,
    260, 259, 294, 294, 295, 260, 261, 260, 295, 295, 296, 261, 262, 261, 296, 296, 297, 262, 302,
    267, 262, 262, 298, 302, 303, 302, 298, 298, 299, 303, 304, 303, 299, 299, 300, 304, 305, 304,
    300, 300, 301, 305, 306, 272, 267, 267, 302, 306, 307, 306, 302, 302, 303, 307, 308, 307, 303,
    303, 304, 308, 309, 308, 304, 304, 305, 309, 310, 277, 272, 272, 306, 310, 311, 310, 306, 306,
    307, 311, 312, 311, 307, 307, 308, 312, 313, 312, 308, 308, 309, 313, 314, 282, 277, 277, 310,
    314, 315, 314, 310, 310, 311, 315, 316, 315, 311, 311, 312, 316, 317, 316, 312, 312, 313, 317,
    318, 287, 282, 282, 314, 318, 319, 318, 314, 314, 315, 319, 320, 319, 315, 315, 316, 320, 321,
    320, 316, 316, 317, 321, 322, 292, 287, 287, 318, 322, 323, 322, 318, 318, 319, 323, 324, 323,
    319, 319, 320, 324, 325, 324, 320, 320, 321, 325, 326, 297, 292, 292, 322, 326, 327, 326, 322,
    322, 323, 327, 328, 327, 323, 323, 324, 328, 329, 328, 324, 324, 325, 329, 298, 262, 297, 297,
    326, 298, 299, 298, 326, 326, 327, 299, 300, 299, 327, 327, 328, 300, 301, 300, 328, 328, 329,
    301, 336, 335, 330, 330, 331, 336, 337, 336, 331, 331, 332, 337, 338, 337, 332, 332, 333, 338,
    339, 338, 333, 333, 334, 339, 341, 340, 335, 335, 336, 341, 342, 341, 336, 336, 337, 342, 343,
    342, 337, 337, 338, 343, 344, 343, 338, 338, 339, 344, 346, 345, 340, 340, 341, 346, 347, 346,
    341, 341, 342, 347, 348, 347, 342, 342, 343, 348, 349, 348, 343, 343, 344, 349, 351, 350, 345,
    345, 346, 351, 352, 351, 346, 346, 347, 352, 353, 352, 347, 347, 348, 353, 354, 353, 348, 348,
    349, 354, 356, 355, 350, 350, 351, 356, 357, 356, 351, 351, 352, 357, 358, 357, 352, 352, 353,
    358, 359, 358, 353, 353, 354, 359, 361, 360, 355, 355, 356, 361, 362, 361, 356, 356, 357, 362,
    363, 362, 357, 357, 358, 363, 364, 363, 358, 358, 359, 364, 366, 365, 360, 360, 361, 366, 367,
    366, 361, 361, 362, 367, 368, 367, 362, 362, 363, 368, 369, 368, 363, 363, 364, 369, 331, 330,
    365, 365, 366, 331, 332, 331, 366, 366, 367, 332, 333, 332, 367, 367, 368, 333, 334, 333, 368,
    368, 369, 334, 374, 339, 334, 334, 370, 374, 375, 374, 370, 370, 371, 375, 376, 375, 371, 371,
    372, 376, 377, 376, 372, 372, 373, 377, 378, 344, 339, 339, 374, 378, 379, 378, 374, 374, 375,
    379, 380, 379, 375, 375, 376, 380, 381, 380, 376, 376, 377, 381, 382, 349, 344, 344, 378, 382,
    383, 382, 378, 378, 379, 383, 384, 383, 379, 379, 380, 384, 385, 384, 380, 380, 381, 385, 386,
    354, 349, 349, 382, 386, 387, 386, 382, 382, 383, 387, 388, 387, 383, 383, 384, 388, 389, 388,
    384, 384, 385, 389, 390, 359, 354, 354, 386, 390, 391, 390, 386, 386, 387, 391, 392, 391, 387,
    387, 388, 392, 393, 392, 388, 388, 389, 393, 394, 364, 359, 359, 390, 394, 395, 394, 390, 390,
    391, 395, 396, 395, 391, 391, 392, 396, 397, 396, 392, 392, 393, 397, 398, 369, 364, 364, 394,
    398, 399, 398, 394, 394, 395, 399, 400, 399, 395, 395, 396, 400, 401, 400, 396, 396, 397, 401,
    370, 334, 369, 369, 398, 370, 371, 370, 398, 398, 399, 371, 372, 371, 399, 399, 400, 372, 373,
    372, 400, 400, 401, 373, 407, 402, 402, 402, 403, 407, 408, 407, 403, 403, 404, 408, 409, 408,
    404, 404, 405, 409, 410, 409, 405, 405, 406, 410, 411, 402, 402, 402, 407, 411, 412, 411, 407,
    407, 408, 412, 413, 412, 408, 408, 409, 413, 414, 413, 409, 409, 410, 414, 415, 402, 402, 402,
    411, 415, 416, 415, 411, 411, 412, 416, 417, 416, 412, 412, 413, 417, 418, 417, 413, 413, 414,
    418, 419, 402, 402, 402, 415, 419, 420, 419, 415, 415, 416, 420, 421, 420, 416, 416, 417, 421,
    422, 421, 417, 417, 418, 422, 423, 402, 402, 402, 419, 423, 424, 423, 419, 419, 420, 424, 425,
    424, 420, 420, 421, 425, 426, 425, 421, 421, 422, 426, 427, 402, 402, 402, 423, 427, 428, 427,
    423, 423, 424, 428, 429, 428, 424, 424, 425, 429, 430, 429, 425, 425, 426, 430, 431, 402, 402,
    402, 427, 431, 432, 431, 427, 427, 428, 432, 433, 432, 428, 428, 429, 433, 434, 433, 429, 429,
    430, 434, 435, 402, 402, 402, 431, 435, 436, 435, 431, 431, 432, 436, 437, 436, 432, 432, 433,
    437, 438, 437, 433, 433, 434, 438, 439, 402, 402, 402, 435, 439, 440, 439, 435, 435, 436, 440,
    441, 440, 436, 436, 437, 441, 442, 441, 437, 437, 438, 442, 443, 402, 402, 402, 439, 443, 444,
    443, 439, 439, 440, 444, 445, 444, 440, 440, 441, 445, 446, 445, 441, 441, 442, 446, 447, 402,
    402, 402, 443, 447, 448, 447, 443, 443, 444, 448, 449, 448, 444, 444, 445, 449, 450, 449, 445,
    445, 446, 450, 451, 402, 402, 402, 447, 451, 452, 451, 447, 447, 448, 452, 453, 452, 448, 448,
    449, 453, 454, 453, 449, 449, 450, 454, 455, 402, 402, 402, 451, 455, 456, 455, 451, 451, 452,
    456, 457, 456, 452, 452, 453, 457, 458, 457, 453, 453, 454, 458, 459, 402, 402, 402, 455, 459,
    460, 459, 455, 455, 456, 460, 461, 460, 456, 456, 457, 461, 462, 461, 457, 457, 458, 462, 463,
    402, 402, 402, 459, 463, 464, 463, 459, 459, 460, 464, 465, 464, 460, 460, 461, 465, 466, 465,
    461, 461, 462, 466, 403, 402, 402, 402, 463, 403, 404, 403, 463, 463, 464, 404, 405, 404, 464,
    464, 465, 405, 406, 405, 465, 465, 466, 406, 471, 410, 406, 406, 467, 471, 472, 471, 467, 467,
    468, 472, 473, 472, 468, 468, 469, 473, 474, 473, 469, 469, 470, 474, 475, 414, 410, 410, 471,
    475, 476, 475, 471, 471, 472, 476, 477, 476, 472, 472, 473, 477, 478, 477, 473, 473, 474, 478,
    479, 418, 414, 414, 475, 479, 480, 479, 475, 475, 476, 480, 481, 480, 476, 476, 477, 481, 482,
    481, 477, 477, 478, 482, 483, 422, 418, 418, 479, 483, 484, 483, 479, 479, 480, 484, 485, 484,
    480, 480, 481, 485, 486, 485, 481, 481, 482, 486, 487, 426, 422, 422, 483, 487, 488, 487, 483,
    483, 484, 488, 489, 488, 484, 484, 485, 489, 490, 489, 485, 485, 486, 490, 491, 430, 426, 426,
    487, 491, 492, 491, 487, 487, 488, 492, 493, 492, 488, 488, 489, 493, 494, 493, 489, 489, 490,
    494, 495, 434, 430, 430, 491, 495, 496, 495, 491, 491, 492, 496, 497, 496, 492, 492, 493, 497,
    498, 497, 493, 493, 494, 498, 499, 438, 434, 434, 495, 499, 500, 499, 495, 495, 496, 500, 501,
    500, 496, 496, 497, 501, 502, 501, 497, 497, 498, 502, 503, 442, 438, 438, 499, 503, 504, 503,
    499, 499, 500, 504, 505, 504, 500, 500, 501, 505, 506, 505, 501, 501, 502, 506, 507, 446, 442,
    442, 503, 507, 508, 507, 503, 503, 504, 508, 509, 508, 504, 504, 505, 509, 510, 509, 505, 505,
    506, 510, 511, 450, 446, 446, 507, 511, 512, 511, 507, 507, 508, 512, 513, 512, 508, 508, 509,
    513, 514, 513, 509, 509, 510, 514, 515, 454, 450, 450, 511, 515, 516, 515, 511, 511, 512, 516,
    517, 516, 512, 512, 513, 517, 518, 517, 513, 513, 514, 518, 519, 458, 454, 454, 515, 519, 520,
    519, 515, 515, 516, 520, 521, 520, 516, 516, 517, 521, 522, 521, 517, 517, 518, 522, 523, 462,
    458, 458, 519, 523, 524, 523, 519, 519, 520, 524, 525, 524, 520, 520, 521, 525, 526, 525, 521,
    521, 522, 526, 527, 466, 462, 462, 523, 527, 528, 527, 523, 523, 524, 528, 529, 528, 524, 524,
    525, 529, 530, 529, 525, 525, 526, 530, 467, 406, 466, 466, 527, 467, 468, 467, 527, 527, 528,
    468, 469, 468, 528, 528, 529, 469, 470, 469, 529, 529, 530, 470u16,
];
