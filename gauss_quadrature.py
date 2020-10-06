

def gauss_quad(f, a, b, n):
    assert b >= a
    
    h = b - a

    if n == 5:
        qq = [0.04691007703066802, 0.23076534494715845, 0.5, 0.7692346550528415, 0.9530899229693319]
        rr = [0.11846344252809454, 0.23931433524968324, 0.28444444444444444, 0.23931433524968324,
              0.11846344252809454]

    if n != 5:
        raise NotImplementedError
            
    return h * sum(r * f(a + h * q) for r, q in zip(rr, qq))


def main():
    
    a, b, c, d = [4, 2, 3, 10]
    f = lambda t: a * t**3 + b * t**2 + c * t + d

    x0, x1 = -3, 2

    I = 0.25 * a * (x1**4-x0**4) + b/3 * (x1**3-x0**3) + 0.5 * c * (x1**2-x0**2) + d * (x1 - x0)

    print(gauss_quad(f, x0, x1, 5))
    print(I)


    return


if __name__ == '__main__':
    main()
