import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_curve(n, seed=None):
    if seed is not None:
        rng = torch.Generator().manual_seed(seed)
    segments = []
    points = [(0, 5)]
    
    for _ in range(n):
        last_point = points[-1]
        x, y = last_point
        
        # Sample a random number for horizontal line length
        line_length = torch.rand(1, generator=rng).squeeze() * 3
        
        # Create the horizontal line
        new_x = x + line_length
        segments.append((x, new_x, y, 0))  # (start_x, end_x, y_value, derivative)

        points.append((new_x, y))
        
        # Randomly decide whether to start a cosine-like curve
        cos_scale = 0.9 + torch.rand(1, generator=rng).item() / 10
        if torch.rand(1, generator=rng).item() < 0.5:
            # Cosine-like curve
            def cos_func(x_val, start_x=new_x, scale=cos_scale, offset=y):
                return scale * (torch.cos(x_val - start_x) - 1) + offset
            
            def cos_derivative(x_val, start_x=new_x, scale=cos_scale):
                return -scale * torch.sin(x_val - start_x)
            
            cos_length = np.pi
        else:
            # Negative cosine-like curve
            def cos_func(x_val, start_x=new_x, scale=cos_scale, offset=y):
                return scale * (1 - torch.cos(x_val - start_x)) + offset
            
            def cos_derivative(x_val, start_x=new_x, scale=cos_scale):
                return scale * torch.sin(x_val - start_x)
            
            cos_length = np.pi
        print((new_x, new_x + cos_length, cos_func, cos_derivative))
        segments.append((new_x, new_x + cos_length, cos_func, cos_derivative))
        new_x = new_x + cos_length
        points.append((new_x, cos_func(new_x)))

    def C(x):
        inp_shape = x.shape
        if len(inp_shape) == 1:
            x = x.unsqueeze(0)
        if len(inp_shape) == 0:
            x = x.unsqueeze(0).unsqueeze(0)
        
        y = torch.zeros(x.shape)
        for segment in segments:
            #print(x, segment[0], segment[1])
            indices = (segment[0] <= x) & (x <= segment[1])
            if callable(segment[2]):
                y[indices] = segment[2](x[indices])
            else:
                y[indices] = segment[2]
        return y.reshape(inp_shape)
    
    def D(x):
        inp_shape = x.shape
        if len(inp_shape) == 1:
            x = x.unsqueeze(0)
        if len(inp_shape) == 0:
            x = x.unsqueeze(0).unsqueeze(0)
        
        y = torch.zeros(x.shape)
        for segment in segments:
            indices = (segment[0] <= x) & (x <= segment[1])
            if callable(segment[3]):
                y[indices] = segment[3](x[indices])
            else:
                y[indices] = segment[3]
        return y.reshape(inp_shape)
    
    max_height_x = max(points, key=lambda p: p[1])[0]
    return C, D, segments[-1][1], max_height_x # Return the curve function, derivative function, and end x value

if __name__ == '__main__':
    # Generate the curve
    n = 2 # Number of iterations
    C, D, last_x, max_hx = generate_curve(n, seed=42)

    # Generate x values for plotting
    x_values = torch.linspace(0, last_x, 1000)
    y_values = [C(x) for x in x_values]
    dy_values = [D(x) for x in x_values]

    # Plot the generated curve and its derivative
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, y_values, label='C(x)', color='b')
    plt.plot(x_values, dy_values, label="C'(x)", color='r')
    plt.title('Procedurally Generated Differentiable Curve and its Derivative')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()
