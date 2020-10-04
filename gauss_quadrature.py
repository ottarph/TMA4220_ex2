

def gauss_quad(f, a, b, n):
    assert b >= a
    
    h = b - a

    if n == 5:
        qq = [0.04691007703066802, 0.23076534494715845, 0.5, 0.7692346550528415, 0.9530899229693319]
        rr = [0.11846344252809454, 0.23931433524968324, 0.28444444444444444, 0.23931433524968324,
              0.11846344252809454]
            
    return sum(r * f(a + h * q) for r, q in zip(rr, qq))


def main():
    
    a, b, c, d = [1, 2, 3, 4]
    f = lambda t: a * t**3 + b * t**2 + c * t + d

    print(gauss_quad(f, 0, 1, 5))


    return


if __name__ == '__main__':
    main()
