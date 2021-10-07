# SEMITONES     INTERVAL        FREQ_RATIO
# 0             Unison          1:1
# 1             Minor 2nd       16:15
# 2             Major 2nd       9:8
# 3             Minor 3rd       6:5
# 4             Major 3rd       5:4
# 5             Perfect 4th     4:3
# 6             Tritone         45:32
# 7             Perfect 5th     3:2
# 8             Minor 6th       8:5
# 9             Major 6th       5:3
# 10            Minor 7th       16:9
# 11            Major 7th       15:8
# 12            Octave          2:1


INTERVALS = {
    '0st': 1.0,
    '1st': 16/15,
    '2st': 9/8,
    '3st': 6/5,
    '4st': 5/4,
    '5st': 4/3,
    '6st': 45/32,
    '7st': 3/2,
    '8st': 8/5,
    '9st': 5/3,
    '10st': 16/9,
    '11st': 15/8,
    '12st': 2.0,
}

perfect_4th = [INTERVALS['0st'], INTERVALS['5st']]
perfect_5th = [INTERVALS['0st'], INTERVALS['7st']]

minor_triad = [
    INTERVALS['0st'],
    INTERVALS['3st'],
    INTERVALS['7st']
]
minor_6th = [
    INTERVALS['0st'],
    INTERVALS['3st'],
    INTERVALS['7st'],
    INTERVALS['9st']
]
minor_7th = [
    INTERVALS['0st'],
    INTERVALS['3st'],
    INTERVALS['7st'],
    INTERVALS['10st']
]
minor_9th = [
    INTERVALS['0st'],
    INTERVALS['3st'],
    INTERVALS['7st'],
    INTERVALS['10st'],
    INTERVALS['12st'] * INTERVALS['2st']
]
minor_6_9th = [
    INTERVALS['0st'],
    INTERVALS['3st'],
    INTERVALS['7st'],
    INTERVALS['9st'],
    INTERVALS['12st'] * INTERVALS['2st']
]
minor_11th = [
    INTERVALS['0st'],
    INTERVALS['3st'],
    INTERVALS['7st'],
    INTERVALS['10st'],
    INTERVALS['12st'] * INTERVALS['2st'],
    INTERVALS['12st'] * INTERVALS['5st']
]

major_triad = [
    INTERVALS['0st'],
    INTERVALS['4st'],
    INTERVALS['7st']
]
major_6th = [
    INTERVALS['0st'],
    INTERVALS['4st'],
    INTERVALS['7st'],
    INTERVALS['9st']
]
major_7th = [
    INTERVALS['0st'],
    INTERVALS['4st'],
    INTERVALS['7st'],
    INTERVALS['11st']
]
major_9th = [
    INTERVALS['0st'],
    INTERVALS['4st'],
    INTERVALS['7st'],
    INTERVALS['11st'],
    INTERVALS['12st'] * INTERVALS['2st']
]
major_6_9th = [
    INTERVALS['0st'],
    INTERVALS['4st'],
    INTERVALS['7st'],
    INTERVALS['9st'],
    INTERVALS['12st'] * INTERVALS['2st']
]
major_11th = [
    INTERVALS['0st'],
    INTERVALS['4st'],
    INTERVALS['7st'],
    INTERVALS['11st'],
    INTERVALS['12st'] * INTERVALS['2st'],
    INTERVALS['12st'] * INTERVALS['5st']
]

dominant_7th = [
    INTERVALS['0st'],
    INTERVALS['4st'],
    INTERVALS['7st'],
    INTERVALS['10st']
]
dominant_9th = [
    INTERVALS['0st'],
    INTERVALS['4st'],
    INTERVALS['7st'],
    INTERVALS['10st'],
    INTERVALS['12st'] * INTERVALS['2st']
]
dominant_11th = [
    INTERVALS['0st'],
    INTERVALS['4st'],
    INTERVALS['7st'],
    INTERVALS['10st'],
    INTERVALS['12st'] * INTERVALS['2st'],
    INTERVALS['12st'] * INTERVALS['5st']
]

augmented_triad = [
    INTERVALS['0st'],
    INTERVALS['4st'],
    INTERVALS['8st'],
]
augmented_7th = [
    INTERVALS['0st'],
    INTERVALS['4st'],
    INTERVALS['8st'],
    INTERVALS['10st'],
]

chords = {
    'perfect_4th': perfect_4th,
    'perfect_5th': perfect_5th,
    'minor_triad': minor_triad,
    'minor_6th': minor_6th,
    'minor_7th': minor_7th,
    'minor_9th': minor_9th,
    'minor_6_9th': minor_6_9th,
    'minor_11th': minor_11th,
    'major_triad': major_triad,
    'major_6th': major_6th,
    'major_7th': major_7th,
    'major_9th': major_9th,
    'major_6_9th': major_6_9th,
    'major_11th': major_11th,
    'dominant_7th': dominant_7th,
    'dominant_9th': dominant_9th,
    'dominant_11th': dominant_11th,
    'augmented_triad': augmented_triad,
    'augmented_7th': augmented_7th
}
