function segmented_img = segmentImage(img, edge_img, radius, fill_threshold)
    % Clear border from edge image
    mask = imclearborder(edge_img);
    % Close disconnected edges
    se = strel("disk", radius); % Morphological structuring element
    %mask= bwmorph(edge_img, "thin", inf);
    mask = imclose(mask, se); % Morphologically close image
    % Fill inside the edges
    mask = imfill(mask, 'holes'); % Fill image holes
    % Remove small objects
    mask = imopen(mask, strel(ones(3,3)));
    mask = bwareaopen(mask, fill_threshold); % Remove filled area under threshold
    segmented_img = img .* uint8(mask);
end

function [original_img, gray_img] = load_image(path)
    % Read File Image
    original_img = imread(path);

    % Convert to grayscale image if original image is RGB
    if size(original_img, 3) > 1
        gray_img = rgb2gray(original_img);
    else
        gray_img = original_img;
    end
end

function BW = apply_edge_detection(img, type, c, laplace_type, log_size, log_std, canny_std, T)
    if strcmp(type, 'Sobel')
        BW = apply_sobel_operator(img, c) >= T;
    elseif strcmp(type, 'Prewitt')
        BW = apply_prewitt_operator(img) >= T;
    elseif strcmp(type, 'Roberts')
        BW = apply_roberts_operator(img) >= T;
    elseif strcmp(type, 'Canny')
        BW = apply_canny_operator(img, canny_std);
    elseif strcmp(type, 'Laplace')
        BW = apply_laplacian_operator(img, laplace_type) >= T;
    elseif strcmp(type, 'LoG')
        BW = apply_log_operator(img, log_size, log_std) >= T;
    end
end

function [Sx, Sy] = get_sobel_mask(c)
    Sx = [-1 0 1; -c 0 c; -1 0 1];
    Sy = [1 c 1; 0 0 0; -1 -c -1];
end

function BW = apply_sobel_operator(img, c)
    [Sx, Sy] = get_sobel_mask(c);
    Jx = conv2(double(img), double(Sx), 'same');
    Jy = conv2(double(img), double(Sy), 'same');
    Jedge = sqrt(Jx.^2 + Jy.^2);
    BW = uint8(Jedge);
end

function [Px, Py] = get_prewitt_mask()
    % Persamaan gradien pada operator Prewitt sama seperti operator Sobel, tetapi menggunakan nilai c = 1:
    [Px, Py] = get_sobel_mask(1);
end

function BW = apply_prewitt_operator(img)
    [Px, Py] = get_prewitt_mask();
    Jx = conv2(double(img), double(Px), 'same');
    Jy = conv2(double(img), double(Py), 'same');
    Jedge = sqrt(Jx.^2 + Jy.^2);
    BW = uint8(Jedge);
end

function [Rx, Ry] = get_roberts_mask()
    Rx = [1 0; 0 -1];
    Ry = [0 1; -1 0];
end

function BW = apply_roberts_operator(img)
    [Rx, Ry] = get_roberts_mask();
    Jx = conv2(double(img), double(Rx), 'same');
    Jy = conv2(double(img), double(Ry), 'same');
    Jedge = sqrt(Jx.^2 + Jy.^2);
    BW = uint8(Jedge);
end

function H = get_laplacian_mask(type)
    if strcmp(type, 'Original')
        % Filter mask used to implement the digital Laplacian (Original)
        H = [0 1 0; 1 -4 1; 0 1 0];
    elseif strcmp(type, 'Diagonal')
        % Filter mask used to implement an extension of this equation that
        % includes the diagonal
        H = [1 1 1; 1 -8 1; 1 1 1];
    end
end

function BW = apply_laplacian_operator(img, type)
    H = get_laplacian_mask(type);
    J = conv2(double(img), double(H), "same");
    BW = uint8(J);
end

function H = get_log_mask(size, std)
    H = fspecial('log', size, std);
end

function BW = apply_log_operator(img, size, std)
    H = get_log_mask(size, std);
    J = conv2(double(img), double(H), "same");
    BW = uint8(J);
end

function BW = apply_canny_operator(img, std)
    BW = edge(img, "canny", [], std);
end