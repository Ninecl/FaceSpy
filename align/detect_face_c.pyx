import numpy as np
import time
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)


cdef np.ndarray[np.float64_t, ndim=3] imresample(np.ndarray[np.uint8_t, ndim=3] img, np.ndarray[np.int_t, ndim=1] sz):
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef int hs = sz[0]
    cdef int ws = sz[1]
    cdef np.float64_t dx = float(w) / ws
    cdef np.float64_t dy = float(h) / hs
    cdef np.ndarray[np.float64_t, ndim=3] im_data = np.zeros((hs, ws, 3))
    for a1 in range(0, hs):
        for a2 in range(0, ws):
            for a3 in range(0, 3):
                im_data[a1, a2, a3] = img[int(a1*dy), int(a2*dx), a3]
    return im_data


cdef np.ndarray[np.float64_t, ndim=2] generateBoundingBox(np.ndarray[np.float32_t, ndim=2] imap, np.ndarray[np.float32_t, ndim=3] reg, np.float64_t scale, float t):
    cdef int stride = 2
    cdef int cellsize = 12
    cdef np.ndarray[np.float32_t, ndim=2] dx1, dy1, dx2, dy2, reg_n
    cdef np.ndarray[np.int64_t, ndim=1] x, y
    cdef np.ndarray[np.float32_t, ndim=1] score
    cdef np.ndarray[np.int64_t, ndim=2] bb
    cdef np.ndarray[np.float64_t, ndim=2] q1, q2, boundingbox
    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:, :, 0])
    dy1 = np.transpose(reg[:, :, 1])
    dx2 = np.transpose(reg[:, :, 2])
    dy2 = np.transpose(reg[:, :, 3])
    y, x = np.where(imap >= t)
    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[(y, x)]
    reg_n = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))
    if reg_n.size == 0:
        reg_n = np.empty((0, 3), dtype=np.float32)
    bb = np.transpose(np.vstack([y, x]))
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize - 1 + 1) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg_n])
    return boundingbox


cdef np.ndarray[np.int16_t, ndim=1] nms(np.ndarray[np.float64_t, ndim=2] boxes, float threshold, char m):
    cdef np.ndarray[np.float64_t, ndim=1] x1, y1, x2, y2, s, area, xx1, yy1, xx2, yy2, w, h, inter, o
    cdef np.ndarray[np.int64_t, ndim=1] I, idx
    cdef np.ndarray[np.int16_t, ndim=1] pick
    cdef int counter
    cdef np.int64_t i
    if boxes.size == 0:
        return np.empty(0, dtype=np.int16)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0: -1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if m == 'M':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)

        I = I[np.where(o <= threshold)]
    pick = pick[0: counter]

    return pick


cdef np.ndarray[np.float64_t, ndim=2] rerec(np.ndarray[np.float64_t, ndim=2] bboxA):
    cdef np.ndarray[np.float64_t, ndim=1] h, w, l
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = np.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2: 4] = bboxA[:, 0: 2] + np.transpose(np.tile(l, (2, 1)))

    return bboxA


cdef np.ndarray[np.int32_t, ndim=2] pad(np.ndarray[np.float64_t, ndim=2] total_boxes, int w, int h):
    cdef np.ndarray[np.int32_t, ndim=1] tmpw, tmph, dx, dy, edx, edy, x, y, ex, ey
    cdef int numbox
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h

    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1
    
    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1

    return np.array([dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph], dtype = np.int32)


cdef np.ndarray[np.int32_t, ndim=2] pad_int32(np.ndarray[np.int32_t, ndim=2] total_boxes, int w, int h):
    cdef np.ndarray[np.int32_t, ndim=1] tmpw, tmph, dx, dy, edx, edy, x, y, ex, ey
    cdef int numbox
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h

    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1
    
    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1
    return np.array([dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph], dtype = np.int32)


cdef np.ndarray[np.float64_t, ndim=2] bbreg(np.ndarray[np.float64_t, ndim=2] boundingbox, np.ndarray[np.float32_t, ndim=2] reg):
    cdef np.ndarray[np.float64_t, ndim=1] w, h, b1, b2, b3, b4

    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))
    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0: 4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox


def detect_face(img, minsize, pnet, rnet, onet, threshold, factor):
    t1 = time.time()
    cdef int factor_count = 0
    cdef np.ndarray[np.float64_t, ndim=2] total_boxes = np.empty((0, 9))
    cdef np.ndarray[np.float32_t, ndim=2] points
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef np.float64_t minl = np.amin([h, w])
    cdef np.float64_t m = 12.0 / minsize
    minl = minl * m
    # cdef np.ndarray[np.float64_t, ndim=1] scales = np.array([])
    scales = []
    while minl >= 12:
        scales += [m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    # first stage
    cdef int hs, ws, first_numbox
    cdef np.ndarray[np.float64_t, ndim=3] im_data
    cdef np.ndarray[np.float64_t, ndim=4] img_x, img_y
    cdef np.ndarray[np.float32_t, ndim=4] out0, out1
    cdef np.ndarray[np.float64_t, ndim=2] boxes
    cdef np.ndarray[np.float64_t, ndim=1] regw, regh, qq1, qq2, qq3, qq4
    cdef np.ndarray[np.int16_t, ndim=1] pick
    cdef np.ndarray[np.int32_t, ndim=1] dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        im_data = imresample(img, np.array([hs, ws]))
        im_data = (im_data - 127.5) * 0.0078125
        img_x = np.expand_dims(im_data, 0)
        img_y = np.transpose(img_x, (0, 2, 1, 3))
        out = pnet(img_y)
        out0 = np.transpose(out[0], (0, 2, 1, 3))
        out1 = np.transpose(out[1], (0, 2, 1, 3))
        boxes = generateBoundingBox(out1[0, :, :, 1].copy(), out0[0, :, :, :].copy(), scale, threshold[0])

        # inter_scale nms
        pick = nms(boxes.copy(), 0.5, 'U')
        # print(pick)
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)
    first_numbox = total_boxes.shape[0]
    if first_numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'U')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0: 4] = np.fix(total_boxes[:, 0: 4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)

    # second stage
    cdef np.ndarray[np.float64_t, ndim=4] tempimg, tempimg1
    cdef np.ndarray[np.float64_t, ndim=3] tmp
    cdef np.ndarray[np.float32_t, ndim=1] score
    cdef np.ndarray[np.float32_t, ndim=2] out2, out3, mv
    second_numbox = total_boxes.shape[0]
    if second_numbox > 0:
        tempimg = np.zeros((24, 24, 3, second_numbox))
        for k in range(0, second_numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k]-1: edy[k], dx[k]-1: edx[k], :] = img[y[k]-1: ey[k], x[k]-1: ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = imresample(tmp.astype(np.uint8), np.array([24, 24]))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
        out = rnet(tempimg1)
        out2 = np.transpose(out[0])
        out3 = np.transpose(out[1])
        score = out3[1, :]
        ipass = np.where(score > threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0], 0: 4].copy(), np.expand_dims(score[ipass].copy(), 1)])
        mv = out2[:, ipass[0]]
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'U')
            total_boxes = total_boxes[pick, :]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
            total_boxes = rerec(total_boxes.copy())
    # third stage
    cdef np.ndarray[np.float32_t, ndim=2] out4, out5, out6
    cdef np.ndarray[np.float64_t, ndim=1] W, H
    cdef np.ndarray[np.int32_t, ndim=2] total_boxes_int32
    numbox = total_boxes.shape[0]
    if numbox > 0:
        total_boxes_int32 = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad_int32(total_boxes_int32.copy(), w, h)
        tempimg = np.zeros((48, 48, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k]-1: edy[k], dx[k]-1: edx[k], :] = img[y[k]-1: ey[k], x[k]-1: ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = imresample(tmp.astype(np.uint8), np.array([48, 48]))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
        out = onet(tempimg1)
        out4 = np.transpose(out[0])
        out5 = np.transpose(out[1])
        out6 = np.transpose(out[2])
        score = out6[1, :]
        points = out5
        ipass = np.where(score > threshold[2])
        points = points[:, ipass[0]]
        total_boxes = np.hstack([total_boxes_int32[ipass[0], 0: 4].copy(), np.expand_dims(score[ipass].copy(), 1)])
        mv = out4[:, ipass[0]]

        W = total_boxes[:, 2] - total_boxes[:, 0] + 1
        H = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[0: 5, :] = np.tile(W, (5, 1)) * points[0: 5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
        points[5: 10, :] = np.tile(H, (5, 1)) * points[5: 10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1
        if total_boxes.shape[0] > 0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
            pick = nms(total_boxes.copy(), 0.7, 'M')
            total_boxes = total_boxes[pick, :]
            points = points[:, pick]
    
    t2 = time.time()
    print(t2 - t1)
    return total_boxes

