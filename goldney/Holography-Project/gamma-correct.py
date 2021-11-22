import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

def GetGammaCurve(image, window_length=1, channels=[0,1,2]):
    # vertically average each column of pixels for designated channels
    img_col_avg = []
    for col in range(len(image[0])):
        col_avg = 0
        for row in range(len(image)):
            col_row_avg = 0
            for channel in channels:
                col_row_avg += image[row][col][channel]
#            col_row_avg = col_row_avg / len(channels)
            col_avg += col_row_avg
        col_avg /= len(image[0])
        img_col_avg.append(col_avg)
#    plt.plot(img_col_avg)
    
    # window average
    ret = []
    if window_length == 1:
        return img_col_avg
    else:
        for end_col in range(1, window_length):
            temp_sum = 0
            for cols in range(0, end_col):
                temp_sum += img_col_avg[cols]
            temp_sum /= end_col
            ret.append(temp_sum)
        for start_col in range(1, len(img_col_avg) - window_length):
            temp_sum = 0
            for cols in range(start_col, start_col + window_length):
                temp_sum += img_col_avg[cols]
            temp_sum /= window_length
            ret.append(temp_sum)
        for start_col in range(len(img_col_avg) - window_length, len(img_col_avg)):
            temp_sum = 0
            for cols in range(start_col, len(img_col_avg)):
                temp_sum += img_col_avg[cols]
            temp_sum /= (len(img_col_avg) - start_col)
            ret.append(temp_sum)
    return ret

def GetCorrectionCurve(in_curve):
    x_list = [0]
    y_list = [0]
    for index in range(len(in_curve)):
        x_list.append(index / len(in_curve) *100)
        y_list.append((in_curve[index] - min(in_curve))/(max(in_curve) - min(in_curve))*100)
    
    # curve fitting
    fit_sigma = np.ones(len(x_list))
    fit_sigma[[0, -1]] = 100
    poly_fit_degree = 4
    poly_fit_coeffs = np.polyfit(x_list, y_list, poly_fit_degree, w=fit_sigma)
    poly_fit_func = np.poly1d(poly_fit_coeffs)
    poly_fit_points = poly_fit_func(x_list)
    max_min_resize_temp = 100 / max (poly_fit_points)
    y_list = np.multiply(y_list, max_min_resize_temp)
    poly_fit_points *= max_min_resize_temp
    print("Fitted function is:\n", poly_fit_func * max_min_resize_temp)
    
    # plotting
    plt.figure(figsize=(10, 8))
    plt.plot(x_list, y_list, label="Measured output")
    plt.plot(x_list, poly_fit_points, label="Fitted Polynomial")
    plt.plot(x_list, x_list, label="Linear Response")
#    plt.plot(poly_fit_points, x_list, label="Gamma correction curve")
    plt.legend(loc="upper left")
    plt.title("Gamma curve with fitted polynomial" )
#    plt.title("Gamma correction curve with polynomial fit of degree " + str(poly_fit_degree))
    plt.xlabel("Input brightness (%)")
    plt.ylabel("Output brightness (%)")
    plt.grid()
    plt.savefig("Gamma_correction.jpg", dpi=300)
    plt.show()



img = cv2.imread('Replay_images/grad.JPG')
raw_curve = GetGammaCurve(img)

GetCorrectionCurve(raw_curve[90:-40])




