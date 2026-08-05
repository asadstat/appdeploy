import os
import io
import tempfile
import warnings
import time
import zipfile

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform

import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)

from shiny import (
    App,
    Inputs,
    Outputs,
    Session,
    reactive,
    render,
    ui
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# TensorFlow settings
# ------------------------------------------------------------

tf.get_logger().setLevel("ERROR")

RANDOM_SEED = 123

np.random.seed(RANDOM_SEED)

tf.random.set_seed(RANDOM_SEED)


# ============================================================
# 2. U-NET MODEL FUNCTION (Replaces CNN)
# ============================================================

def create_unet_model(
    patch_size,
    n_bands,
    n_classes,
    learning_rate=0.001
):
    inputs = tf.keras.layers.Input(shape=(patch_size, patch_size, n_bands))

    # Encoder
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c1)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(p1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c2)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c3)
    c3 = tf.keras.layers.BatchNormalization()(c3)

    # Decoder
    u2 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c3)
    u2 = tf.keras.layers.concatenate([u2, c2])
    c4 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u2)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c4)
    c4 = tf.keras.layers.BatchNormalization()(c4)

    u1 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c4)
    u1 = tf.keras.layers.concatenate([u1, c1])
    c5 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(u1)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c5)
    c5 = tf.keras.layers.BatchNormalization()(c5)

    # Output mapping per pixel, pooled globally to match patch-label training setup
    x = tf.keras.layers.GlobalAveragePooling2D()(c5)
    x = tf.keras.layers.Dropout(0.40)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 3. READ SATELLITE IMAGE
# ============================================================

def read_satellite_image(
    raster_path
):

    with rasterio.open(
        raster_path
    ) as source:

        image = source.read()

        image = np.moveaxis(
            image,
            0,
            -1
        )

        profile = source.profile.copy()

        transform_data = source.transform

        raster_crs = source.crs

        bounds = source.bounds

        height = source.height

        width = source.width

        band_count = source.count

        resolution = source.res

        nodata_value = source.nodata

    image = image.astype(
        np.float32
    )

    if nodata_value is not None:

        image[
            image == nodata_value
        ] = np.nan

    return {

        "image": image,

        "profile": profile,

        "transform": transform_data,

        "crs": raster_crs,

        "bounds": bounds,

        "height": height,

        "width": width,

        "bands": band_count,

        "resolution": resolution
    }


# ============================================================
# 6. NORMALIZE PATCHES
# ============================================================

def calculate_normalization(

    X

):

    band_mean = np.mean(

        X,

        axis=(
            0,
            1,
            2
        )

    )

    band_standard_deviation = np.std(

        X,

        axis=(
            0,
            1,
            2
        )

    )

    band_standard_deviation[

        band_standard_deviation == 0

    ] = 1

    X_normalized = (

        X -

        band_mean.reshape(

            1,
            1,
            1,
            -1

        )

    ) / (

        band_standard_deviation.reshape(

            1,
            1,
            1,
            -1

        )

    )

    return (

        X_normalized.astype(
            np.float32
        ),

        band_mean.astype(
            np.float32
        ),

        band_standard_deviation.astype(
            np.float32
        )
    )


# ============================================================
# 7. CREATE FULL-IMAGE PATCHES
# ============================================================

def classify_complete_image(

    image,

    model,

    patch_size,

    band_mean,

    band_standard_deviation,

    prediction_batch_size=128,
    progress=None

):

    image_height = image.shape[0]

    image_width = image.shape[1]

    number_bands = image.shape[2]

    half_patch = (

        patch_size // 2

    )

    classified_image = np.full(

        (
            image_height,
            image_width
        ),

        np.nan,

        dtype=np.float32

    )

    patch_batch = []

    location_batch = []
    
    total_rows = image_height - (2 * half_patch)
    current_row_count = 0
    start_time = time.time()

    def predict_current_batch():

        nonlocal patch_batch

        nonlocal location_batch

        if len(
            patch_batch
        ) == 0:

            return

        X_batch = np.stack(

            patch_batch,

            axis=0

        ).astype(

            np.float32

        )

        X_batch = (

            X_batch -

            band_mean.reshape(

                1,
                1,
                1,
                -1

            )

        ) / (

            band_standard_deviation.reshape(

                1,
                1,
                1,
                -1

            )

        )

        probability = model.predict(

            X_batch,

            batch_size=
                prediction_batch_size,

            verbose=0

        )

        predicted_class = (

            np.argmax(

                probability,

                axis=1

            ) + 1

        )

        for current_index in range(

            len(
                location_batch
            )

        ):

            current_row, current_column = (

                location_batch[
                    current_index
                ]

            )

            classified_image[

                current_row,

                current_column

            ] = (

                predicted_class[
                    current_index
                ]

            )

        patch_batch = []

        location_batch = []

    for current_row in range(

        half_patch,

        image_height -
        half_patch

    ):

        current_row_count += 1

        for current_column in range(

            half_patch,

            image_width -
            half_patch

        ):

            row_start = (

                current_row -
                half_patch

            )

            row_end = (

                current_row +
                half_patch

            )

            column_start = (

                current_column -
                half_patch

            )

            column_end = (

                current_column +
                half_patch

            )

            current_patch = image[

                row_start:row_end,

                column_start:column_end,

                :

            ]

            if np.isnan(

                current_patch

            ).any():

                continue

            patch_batch.append(

                current_patch

            )

            location_batch.append(

                (
                    current_row,
                    current_column
                )

            )

            if len(
                patch_batch
            ) >= prediction_batch_size:

                predict_current_batch()

        if progress is not None and total_rows > 0:
            fraction = current_row_count / total_rows
            percent_complete = fraction * 100
            percent_remaining = max(0.0, 100.0 - percent_complete)
            
            elapsed = time.time() - start_time
            remaining = max(0.0, (elapsed / fraction) - elapsed) if fraction > 0 else 0.0

            progress.set(
                value=fraction,
                message="Running Land Cover Classification...",
                detail=(
                    f"Complete: {percent_complete:.1f}% | "
                    f"Remaining: {percent_remaining:.1f}% | "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"Remaining Time: {remaining:.1f}s"
                )
            )

    predict_current_batch()

    return classified_image


# ============================================================
# 8. CALCULATE PIXEL AREA
# ============================================================

def calculate_pixel_area_hectare(

    raster_profile

):

    raster_crs = (

        raster_profile[
            "crs"
        ]

    )

    raster_transform = (

        raster_profile[
            "transform"
        ]

    )

    pixel_width = abs(

        raster_transform.a

    )

    pixel_height = abs(

        raster_transform.e

    )

    if (

        raster_crs is not None

        and

        raster_crs.is_projected

    ):

        pixel_area_square_meter = (

            pixel_width *

            pixel_height

        )

        pixel_area_hectare = (

            pixel_area_square_meter /

            10000

        )

    else:

        pixel_area_hectare = np.nan

    return pixel_area_hectare


# ============================================================
# 9. CALCULATE LAND-COVER AREA
# ============================================================

def calculate_land_cover_area(

    classified_image,

    class_names,

    raster_profile

):

    pixel_area_hectare = (

        calculate_pixel_area_hectare(

            raster_profile

        )

    )

    result_list = []

    valid_values = (

        classified_image[

            ~np.isnan(
                classified_image
            )

        ]

        .astype(int)

    )

    total_pixels = (

        len(
            valid_values
        )

    )

    for class_id in range(

        1,

        len(
            class_names
        ) + 1

    ):

        pixel_count = int(

            np.sum(

                valid_values

                ==

                class_id

            )

        )

        if np.isnan(

            pixel_area_hectare

        ):

            area_hectare = np.nan

        else:

            area_hectare = (

                pixel_count *

                pixel_area_hectare

            )

        if total_pixels > 0:

            percentage = (

                pixel_count /

                total_pixels *

                100

            )

        else:

            percentage = 0

        result_list.append({

            "Class_ID":
                class_id,

            "Land_Cover":
                class_names[
                    class_id - 1
                ],

            "Pixel_Count":
                pixel_count,

            "Area_ha":
                area_hectare,

            "Percentage":
                percentage
        })

    area_table = pd.DataFrame(

        result_list

    )

    return area_table


# ============================================================
# 10. CREATE CLASSIFIED MAP
# ============================================================

def create_classification_figure(

    classified_image,

    class_names

):

    number_classes = len(

        class_names

    )

    base_colors = [

        "#2166ac",

        "#1b7837",

        "#f6e620",

        "#d73027",

        "#8c6d31",

        "#7b3294",

        "#00a6a6",

        "#e66101"

    ]

    selected_colors = (

        base_colors[
            :number_classes
        ]

    )

    color_map = (

        ListedColormap(

            selected_colors

        )

    )

    figure, axis = (

        plt.subplots(

            figsize=(
                10,
                8
            )

        )

    )

    image_plot = (

        axis.imshow(

            classified_image,

            cmap=color_map,

            vmin=1,

            vmax=number_classes

        )

    )

    color_bar = (

        figure.colorbar(

            image_plot,

            ax=axis,

            ticks=np.arange(

                1,

                number_classes + 1

            )

        )

    )

    color_bar.ax.set_yticklabels(

        class_names

    )

    axis.set_title(

        "U-Net Based Land Cover Classification",

        fontsize=15,

        fontweight="bold"

    )

    axis.set_xlabel(

        "Column"

    )

    axis.set_ylabel(

        "Row"

    )

    figure.tight_layout()

    return figure


# ============================================================
# 11. USER INTERFACE
# ============================================================

app_ui = ui.page_navbar(

    ui.nav_panel(

        "Home",

        ui.h2(

            "DeepLand-UNET"

        ),

        ui.p(

            "U-Net Based Satellite Land Cover "
            "Classification System"

        ),

        ui.layout_columns(

            ui.card(

                ui.card_header(
                    "Model"
                ),

                ui.h3(
                    "U-Net"
                ),

                ui.p(
                    "U-Net Convolutional Architecture"
                )

            ),

            ui.card(

                ui.card_header(
                    "Input"
                ),

                ui.h3(
                    "Satellite GeoTIFF"
                ),

                ui.p(
                    "Multiband raster image"
                )

            ),

            ui.card(

                ui.card_header(
                    "Training Data"
                ),

                ui.h3(
                    "Zip Archives"
                ),

                ui.p(
                    "Multiple class zip files"
                )

            ),

            ui.card(

                ui.card_header(
                    "Output"
                ),

                ui.h3(
                    "Land Cover Map"
                ),

                ui.p(
                    "Classified GeoTIFF"
                )

            ),

            col_widths=(
                3,
                3,
                3,
                3
            )

        ),

        ui.hr(),

        ui.h4(
            "System Workflow"
        ),

        ui.tags.ol(

            ui.tags.li(

                "Upload multiband satellite GeoTIFF"

            ),

            ui.tags.li(

                "Upload multiple class zip files (zip name = class label, containing shapefile components)"

            ),

            ui.tags.li(

                "Extract image patches"

            ),

            ui.tags.li(

                "Train U-Net model"

            ),

            ui.tags.li(

                "Run full-image classification"

            ),

            ui.tags.li(

                "Calculate accuracy and land-cover area"

            ),

            ui.tags.li(

                "Download GeoTIFF and CSV results"

            )
        )
    ),


    # ========================================================
    # DATA INPUT TAB
    # ========================================================

    ui.nav_panel(

        "Data Input",

        ui.layout_sidebar(

            ui.sidebar(

                ui.input_file(

                    "raster_file",

                    "Upload Multiband GeoTIFF",

                    accept=[
                        ".tif",
                        ".tiff"
                    ],

                    multiple=False

                ),

                ui.input_file(

                    "training_zips",

                    "Upload Class Zip Files",

                    accept=[
                        ".zip"
                    ],

                    multiple=True

                ),

                ui.input_action_button(

                    "load_data",

                    "Load Data",

                    class_="btn-primary"

                ),

                ui.hr(),

                ui.p(

                    "Zip naming guide:"

                ),

                ui.p(
                    "Upload separate zip files for each class (e.g., Water.zip, Vegetation.zip). Each zip should contain vector shapefile elements (.shp, .shx, .dbf) or a .geojson file."
                )

            ),

            ui.card(

                ui.card_header(

                    "Satellite Image Information"

                ),

                ui.output_text_verbatim(

                    "raster_information"

                )
            ),

            ui.br(),

            ui.card(

                ui.card_header(

                    "Training Data Summary"

                ),

                ui.output_data_frame(

                    "training_table"

                )
            )
        )
    ),


    # ========================================================
    # U-NET SETTINGS TAB
    # ========================================================

    ui.nav_panel(

        "U-Net Settings",

        ui.layout_sidebar(

            ui.sidebar(

                ui.input_select(

                    "patch_size",

                    "Patch Size",

                    choices={

                        "16":
                            "16 × 16",

                        "32":
                            "32 × 32",

                        "64":
                            "64 × 64"

                    },

                    selected="32"

                ),

                ui.input_numeric(

                    "epochs",

                    "Epochs",

                    value=30,

                    min=1,

                    max=500

                ),

                ui.input_numeric(

                    "batch_size",

                    "Training Batch Size",

                    value=32,

                    min=1

                ),

                ui.input_numeric(

                    "learning_rate",

                    "Learning Rate",

                    value=0.001,

                    min=0.000001,

                    max=0.1,

                    step=0.0001

                ),

                ui.input_slider(

                    "validation_split",

                    "Validation Proportion",

                    min=0.10,

                    max=0.50,

                    value=0.20,

                    step=0.05

                ),

                ui.input_action_button(

                    "train_model",

                    "Train U-Net Model",

                    class_="btn-success"

                )

            ),

            ui.card(

                ui.card_header(

                    "U-Net Model and Training Status"

                ),

                ui.output_text_verbatim(

                    "training_status"

                )
            )
        )
    ),


    # ========================================================
    # TRAINING RESULTS TAB
    # ========================================================

    ui.nav_panel(

        "Training Results",

        ui.layout_columns(

            ui.card(

                ui.card_header(

                    "U-Net Accuracy"

                ),

                ui.output_plot(

                    "accuracy_plot",

                    height="450px"

                )
            ),

            ui.card(

                ui.card_header(

                    "U-Net Loss"

                ),

                ui.output_plot(

                    "loss_plot",

                    height="450px"

                )
            ),

            col_widths=(
                6,
                6
            )
        )
    ),


    # ========================================================
    # CLASSIFICATION TAB
    # ========================================================

    ui.nav_panel(

        "Classification",

        ui.layout_sidebar(

            ui.sidebar(

                ui.input_numeric(

                    "prediction_batch_size",

                    "Prediction Batch Size",

                    value=128,

                    min=1

                ),

                ui.input_action_button(

                    "classify_image",

                    "Run Land Cover Classification",

                    class_="btn-danger"

                ),

                ui.hr(),

                ui.p(

                    "Large images require substantial "
                    "RAM and processing time."

                )

            ),

            ui.card(

                ui.card_header(

                    "U-Net Classified Land Cover Map"

                ),

                ui.output_plot(

                    "classified_map",

                    height="750px"

                )
            )
        )
    ),


    # ========================================================
    # ACCURACY TAB
    # ========================================================

    ui.nav_panel(

        "Accuracy Assessment",

        ui.layout_columns(

            ui.card(

                ui.card_header(

                    "Overall Accuracy"

                ),

                ui.output_text(

                    "overall_accuracy"

                )
            ),

            ui.card(

                ui.card_header(

                    "Kappa Coefficient"

                ),

                ui.output_text(

                    "kappa_value"

                )
            ),

            ui.card(

                ui.card_header(

                    "Validation Samples"

                ),

                ui.output_text(

                    "validation_samples"

                )
            ),

            col_widths=(
                4,
                4,
                4
            )
        ),

        ui.br(),

        ui.card(

            ui.card_header(

                "Confusion Matrix"

            ),

            ui.output_data_frame(

                "confusion_matrix_table"

            )
        ),

        ui.br(),

        ui.card(

            ui.card_header(

                "Classification Report"

            ),

            ui.output_data_frame(

                "classification_report_table"

            )
        )
    ),


    # ========================================================
    # AREA TAB
    # ========================================================

    ui.nav_panel(

        "Land Cover Area",

        ui.layout_columns(

            ui.card(

                ui.card_header(

                    "Land Cover Area Statistics"

                ),

                ui.output_data_frame(

                    "area_table"

                )
            ),

            ui.card(

                ui.card_header(

                    "Land Cover Area Chart"

                ),

                ui.output_plot(

                    "area_plot",

                    height="450px"

                )
            ),

            col_widths=(
                7,
                5
            )
        )
    ),


    # ========================================================
    # DOWNLOAD TAB
    # ========================================================

    ui.nav_panel(

        "Download Results",

        ui.h3(

            "Download U-Net Classification Results"

        ),

        ui.download_button(

            "download_raster",

            "Download Classified GeoTIFF",

            class_="btn-primary"

        ),

        ui.br(),

        ui.br(),

        ui.download_button(

            "download_area",

            "Download Land Cover Area CSV",

            class_="btn-success"

        ),

        ui.br(),

        ui.br(),

        ui.download_button(

            "download_accuracy",

            "Download Accuracy Results CSV",

            class_="btn-warning"

        )
    ),

    title="DeepLand-UNET",

    id="main_navigation"
)


# ============================================================
# 12. SERVER
# ============================================================

def server(

    input: Inputs,

    output: Outputs,

    session: Session

):

    raster_data = reactive.Value(None)
    training_dataframe = reactive.Value(None)
    extracted_patches_data = reactive.Value(None)
    unet_model = reactive.Value(None)
    unet_history = reactive.Value(None)
    class_names = reactive.Value(None)
    normalization_data = reactive.Value(None)
    accuracy_data = reactive.Value(None)
    classified_data = reactive.Value(None)
    area_data = reactive.Value(None)


    # --------------------------------------------------------
    # LOAD DATA FROM ZIP ARCHIVES
    # --------------------------------------------------------

    @reactive.effect

    @reactive.event(

        input.load_data

    )

    def load_input_data():

        raster_upload = (

            input.raster_file()

        )

        training_zips = (

            input.training_zips()

        )

        if (

            raster_upload is None

            or

            training_zips is None

            or

            len(training_zips) == 0

        ):

            ui.notification_show(

                "Please upload both a GeoTIFF and at least one zip file per class.",

                type="error"

            )

            return

        try:
            with ui.Progress(min=0, max=1) as progress:
                progress.set(value=0, message="Loading Satellite Image...", detail="Reading raster file...")
                
                current_raster = (
                    read_satellite_image(
                        raster_upload[0][
                            "datapath"
                        ]
                    )
                )
                
                progress.set(value=0.3, message="Parsing Zip Files...", detail="Extracting spatial vector data...")
                
                patch_list = []
                class_list = []
                summary_records = []
                
                image = current_raster["image"]
                transform_data = current_raster["transform"]
                raster_crs = current_raster["crs"]
                image_height = image.shape[0]
                image_width = image.shape[1]
                number_bands = image.shape[2]
                patch_size = int(input.patch_size())
                half_patch = patch_size // 2

                for zip_info in training_zips:
                    orig_name = zip_info["name"]
                    class_label = os.path.splitext(os.path.basename(orig_name))[0]
                    
                    count_in_zip = 0
                    with tempfile.TemporaryDirectory() as tmpdir:
                        with zipfile.ZipFile(zip_info["datapath"], "r") as z:
                            z.extractall(tmpdir)
                        
                        vector_file_path = None
                        for root, dirs, files in os.walk(tmpdir):
                            for file in files:
                                if file.lower().endswith(".shp") or file.lower().endswith(".geojson"):
                                    vector_file_path = os.path.join(root, file)
                                    break
                            if vector_file_path:
                                break
                                
                        if not vector_file_path:
                            summary_records.append({"Class": class_label, "Valid_Patches": 0})
                            continue
                            
                        gdf = gpd.read_file(vector_file_path)
                        if gdf.empty:
                            summary_records.append({"Class": class_label, "Valid_Patches": 0})
                            continue
                            
                        if raster_crs is not None and gdf.crs is not None:
                            if gdf.crs != raster_crs:
                                gdf = gdf.to_crs(raster_crs)
                        elif raster_crs is not None and gdf.crs is None:
                            gdf.set_crs("EPSG:4326", inplace=True)
                            gdf = gdf.to_crs(raster_crs)

                        for geom in gdf.geometry:
                            if geom is None:
                                continue
                            
                            if geom.geom_type == 'Point':
                                x_coords, y_coords = [geom.x], [geom.y]
                            elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                                centroid = geom.centroid
                                x_coords, y_coords = [centroid.x], [centroid.y]
                            else:
                                continue

                            for xi, yi in zip(x_coords, y_coords):
                                try:
                                    row_number, column_number = rowcol(
                                        transform_data,
                                        xi,
                                        yi
                                    )
                                    row_number = int(row_number)
                                    column_number = int(column_number)
                                except Exception:
                                    continue
                                    
                                row_start = row_number - half_patch
                                row_end = row_number + half_patch
                                column_start = column_number - half_patch
                                column_end = column_number + half_patch
                                
                                if row_start < 0 or row_end > image_height or column_start < 0 or column_end > image_width:
                                    continue
                                    
                                current_patch = image[row_start:row_end, column_start:column_end, :]
                                if current_patch.shape != (patch_size, patch_size, number_bands) or np.isnan(current_patch).any():
                                    continue
                                    
                                patch_list.append(current_patch)
                                class_list.append(class_label)
                                count_in_zip += 1

                    summary_records.append({"Class": class_label, "Valid_Patches": count_in_zip})

                if len(patch_list) == 0:
                    raise ValueError("No valid patches could be extracted from the uploaded zip files.")

                X = np.stack(patch_list, axis=0).astype(np.float32)
                y_text = np.asarray(class_list)

                raster_data.set(current_raster)
                training_dataframe.set(pd.DataFrame(summary_records))
                extracted_patches_data.set({"X": X, "y_text": y_text})

            ui.notification_show(
                "Satellite image and zip training archives loaded successfully.",
                type="message",
                duration=5
            )

        except Exception as error:
            ui.notification_show(
                str(error),
                type="error",
                duration=10
            )


    # --------------------------------------------------------
    # RASTER INFORMATION
    # --------------------------------------------------------

    @render.text

    def raster_information():

        current_raster = raster_data.get()

        if current_raster is None:
            return "No satellite image loaded."

        return (
            f"Number of Bands: {current_raster['bands']}\n"
            f"Rows: {current_raster['height']}\n"
            f"Columns: {current_raster['width']}\n"
            f"Resolution: {current_raster['resolution']}\n"
            f"CRS: {current_raster['crs']}\n"
            f"Bounds: {current_raster['bounds']}"
        )


    # --------------------------------------------------------
    # TRAINING TABLE
    # --------------------------------------------------------

    @render.data_frame

    def training_table():

        current_data = training_dataframe.get()

        if current_data is None:
            return render.DataGrid(pd.DataFrame())

        return render.DataGrid(current_data, filters=True, height="400px")


    # --------------------------------------------------------
    # TRAIN U-NET MODEL
    # --------------------------------------------------------

    @reactive.effect

    @reactive.event(

        input.train_model

    )

    def train_unet_model():

        current_raster = raster_data.get()
        patches_payload = extracted_patches_data.get()

        if current_raster is None or patches_payload is None:
            ui.notification_show(
                "Load the satellite image and class zip files first.",
                type="error"
            )
            return

        try:
            with ui.Progress(min=0, max=1) as progress:
                progress.set(value=0, message="Initializing Training...", detail="Preparing dataset...")
                start_time = time.time()

                X = patches_payload["X"]
                y_text = patches_payload["y_text"]

                unique_classes = sorted(np.unique(y_text).tolist())

                if len(unique_classes) < 2:
                    raise ValueError("At least two land-cover classes are required.")

                class_to_id = {class_name: class_number for class_number, class_name in enumerate(unique_classes)}
                y = np.asarray([class_to_id[current_class] for current_class in y_text])

                X_normalized, band_mean, band_sd = calculate_normalization(X)
                class_count = np.bincount(y)
                use_stratification = np.min(class_count) >= 2

                X_train, X_validation, y_train, y_validation = train_test_split(
                    X_normalized,
                    y,
                    test_size=input.validation_split(),
                    random_state=RANDOM_SEED,
                    stratify=(y if use_stratification else None)
                )

                model = create_unet_model(
                    patch_size=int(input.patch_size()),
                    n_bands=current_raster["bands"],
                    n_classes=len(unique_classes),
                    learning_rate=input.learning_rate()
                )

                total_epochs = int(input.epochs())

                class CustomCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        current_epoch = epoch + 1
                        fraction = current_epoch / total_epochs
                        pct_complete = fraction * 100
                        pct_rem = max(0.0, 100.0 - pct_complete)
                        
                        elapsed = time.time() - start_time
                        remaining = max(0.0, (elapsed / fraction) - elapsed) if fraction > 0 else 0.0

                        progress.set(
                            value=fraction,
                            message=f"Training Epoch {current_epoch}/{total_epochs}",
                            detail=(
                                f"Complete: {pct_complete:.1f}% | "
                                f"Remaining: {pct_rem:.1f}% | "
                                f"Elapsed: {elapsed:.1f}s | "
                                f"Remaining Time: {remaining:.1f}s"
                            )
                        )

                callbacks = [
                    CustomCallback(),
                    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
                ]

                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_validation, y_validation),
                    epochs=total_epochs,
                    batch_size=int(input.batch_size()),
                    callbacks=callbacks,
                    verbose=0
                )

                probability = model.predict(X_validation, verbose=0)
                predicted_class = np.argmax(probability, axis=1)

                overall_accuracy = accuracy_score(y_validation, predicted_class)
                kappa = cohen_kappa_score(y_validation, predicted_class)
                matrix = confusion_matrix(y_validation, predicted_class, labels=np.arange(len(unique_classes)))
                matrix_dataframe = pd.DataFrame(matrix, index=unique_classes, columns=unique_classes)

                report = classification_report(
                    y_validation,
                    predicted_class,
                    labels=np.arange(len(unique_classes)),
                    target_names=unique_classes,
                    output_dict=True,
                    zero_division=0
                )
                report_dataframe = pd.DataFrame(report).transpose()

                unet_model.set(model)
                unet_history.set(history.history)
                class_names.set(unique_classes)
                normalization_data.set({"mean": band_mean, "sd": band_sd})
                accuracy_data.set({
                    "overall_accuracy": overall_accuracy,
                    "kappa": kappa,
                    "validation_samples": len(y_validation),
                    "confusion_matrix": matrix_dataframe,
                    "classification_report": report_dataframe
                })

            ui.notification_show(
                f"U-Net training completed. Validation accuracy: {overall_accuracy * 100:.2f}%",
                type="message",
                duration=10
            )

        except Exception as error:
            ui.notification_show(
                str(error),
                type="error",
                duration=15
            )


    # --------------------------------------------------------
    # TRAINING STATUS
    # --------------------------------------------------------

    @render.text

    def training_status():

        current_history = unet_history.get()
        current_model = unet_model.get()

        if current_history is None or current_model is None:
            return "U-Net model has not been trained."

        final_training_accuracy = current_history["accuracy"][-1]
        final_validation_accuracy = current_history["val_accuracy"][-1]

        model_summary = []
        current_model.summary(print_fn=lambda line: model_summary.append(line))

        return (
            "U-NET TRAINING COMPLETED\n\n"
            f"Final Training Accuracy: {final_training_accuracy * 100:.2f}%\n"
            f"Final Validation Accuracy: {final_validation_accuracy * 100:.2f}%\n\n"
            "U-NET MODEL SUMMARY\n" + "\n".join(model_summary)
        )


    # --------------------------------------------------------
    # ACCURACY PLOT
    # --------------------------------------------------------

    @render.plot

    def accuracy_plot():

        current_history = unet_history.get()
        figure, axis = plt.subplots(figsize=(8, 5))

        if current_history is None:
            axis.text(0.5, 0.5, "Train the U-Net model to display accuracy.", ha="center", va="center", fontsize=13)
            axis.set_axis_off()
            return figure

        epoch_number = np.arange(1, len(current_history["accuracy"]) + 1)
        axis.plot(epoch_number, current_history["accuracy"], label="Training Accuracy")
        axis.plot(epoch_number, current_history["val_accuracy"], label="Validation Accuracy")
        axis.set_title("U-Net Training and Validation Accuracy")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Accuracy")
        axis.legend()
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        return figure


    # --------------------------------------------------------
    # LOSS PLOT
    # --------------------------------------------------------

    @render.plot

    def loss_plot():

        current_history = unet_history.get()
        figure, axis = plt.subplots(figsize=(8, 5))

        if current_history is None:
            axis.text(0.5, 0.5, "Train the U-Net model to display loss.", ha="center", va="center", fontsize=13)
            axis.set_axis_off()
            return figure

        epoch_number = np.arange(1, len(current_history["loss"]) + 1)
        axis.plot(epoch_number, current_history["loss"], label="Training Loss")
        axis.plot(epoch_number, current_history["val_loss"], label="Validation Loss")
        axis.set_title("U-Net Training and Validation Loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.legend()
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        return figure


    # --------------------------------------------------------
    # RUN FULL-IMAGE CLASSIFICATION
    # --------------------------------------------------------

    @reactive.effect

    @reactive.event(

        input.classify_image

    )

    def run_classification():

        current_raster = raster_data.get()
        current_model = unet_model.get()
        current_normalization = normalization_data.get()
        current_classes = class_names.get()

        if current_raster is None or current_model is None or current_normalization is None or current_classes is None:
            ui.notification_show("Train the U-Net model first.", type="error")
            return

        try:
            with ui.Progress(min=0, max=1) as progress:
                progress.set(value=0, message="Initializing Classification...", detail="Starting row processing...")
                
                classified_image = classify_complete_image(
                    image=current_raster["image"],
                    model=current_model,
                    patch_size=int(input.patch_size()),
                    band_mean=current_normalization["mean"],
                    band_standard_deviation=current_normalization["sd"],
                    prediction_batch_size=int(input.prediction_batch_size()),
                    progress=progress
                )

            output_profile = current_raster["profile"].copy()
            output_profile.update({
                "count": 1,
                "dtype": "uint16",
                "nodata": 0,
                "compress": "lzw"
            })

            output_raster = np.where(np.isnan(classified_image), 0, classified_image).astype(np.uint16)
            area_table_data = calculate_land_cover_area(
                classified_image=classified_image,
                class_names=current_classes,
                raster_profile=output_profile
            )

            classified_data.set({
                "image": classified_image,
                "output_image": output_raster,
                "profile": output_profile
            })

            area_data.set(area_table_data)

            ui.notification_show("U-Net land-cover classification completed.", type="message", duration=10)

        except Exception as error:
            ui.notification_show(str(error), type="error", duration=15)


    # --------------------------------------------------------
    # CLASSIFIED MAP
    # --------------------------------------------------------

    @render.plot

    def classified_map():

        current_result = classified_data.get()
        current_classes = class_names.get()

        if current_result is None or current_classes is None:
            figure, axis = plt.subplots(figsize=(10, 8))
            axis.text(0.5, 0.5, "Run land-cover classification to display the map.", ha="center", va="center", fontsize=14)
            axis.set_axis_off()
            return figure

        return create_classification_figure(current_result["image"], current_classes)


    # --------------------------------------------------------
    # ACCURACY OUTPUTS
    # --------------------------------------------------------

    @render.text

    def overall_accuracy():

        current_accuracy = accuracy_data.get()
        if current_accuracy is None:
            return "Not available"
        return f"{current_accuracy['overall_accuracy'] * 100:.2f}%"


    @render.text

    def kappa_value():

        current_accuracy = accuracy_data.get()
        if current_accuracy is None:
            return "Not available"
        return f"{current_accuracy['kappa']:.4f}"


    @render.text

    def validation_samples():

        current_accuracy = accuracy_data.get()
        if current_accuracy is None:
            return "Not available"
        return str(current_accuracy["validation_samples"])


    @render.data_frame

    def confusion_matrix_table():

        current_accuracy = accuracy_data.get()
        if current_accuracy is None:
            return render.DataGrid(pd.DataFrame())

        matrix = current_accuracy["confusion_matrix"]
        matrix.index.name = "Actual Class"
        return render.DataGrid(matrix.reset_index(), filters=True, height="450px")


    @render.data_frame

    def classification_report_table():

        current_accuracy = accuracy_data.get()
        if current_accuracy is None:
            return render.DataGrid(pd.DataFrame())

        report = current_accuracy["classification_report"].reset_index().rename(columns={"index": "Class"})
        return render.DataGrid(report, filters=True, height="500px")


    # --------------------------------------------------------
    # AREA TABLE
    # --------------------------------------------------------

    @render.data_frame

    def area_table():

        current_area = area_data.get()
        if current_area is None:
            return render.DataGrid(pd.DataFrame())

        display_area = current_area.copy()
        display_area["Area_ha"] = display_area["Area_ha"].round(2)
        display_area["Percentage"] = display_area["Percentage"].round(2)
        return render.DataGrid(display_area, filters=True, height="450px")


    # --------------------------------------------------------
    # AREA CHART
    # --------------------------------------------------------

    @render.plot

    def area_plot():

        current_area = area_data.get()
        figure, axis = plt.subplots(figsize=(8, 6))

        if current_area is None:
            axis.text(0.5, 0.5, "Run classification to calculate land-cover area.", ha="center", va="center", fontsize=13)
            axis.set_axis_off()
            return figure

        if current_area["Area_ha"].isna().all():
            value_column = "Pixel_Count"
            y_label = "Pixel Count"
        else:
            value_column = "Area_ha"
            y_label = "Area (Hectare)"

        axis.bar(current_area["Land_Cover"], current_area[value_column])
        axis.set_title("Land Cover Area")
        axis.set_xlabel("Land Cover Class")
        axis.set_ylabel(y_label)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        return figure


    # --------------------------------------------------------
    # DOWNLOAD CLASSIFIED GEOTIFF
    # --------------------------------------------------------

    @render.download(filename="UNet_Land_Cover_Classification.tif")

    def download_raster():

        current_result = classified_data.get()
        if current_result is None:
            return

        temporary_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        temporary_file.close()

        with rasterio.open(temporary_file.name, "w", **current_result["profile"]) as destination:
            destination.write(current_result["output_image"], 1)

        return temporary_file.name


    # --------------------------------------------------------
    # DOWNLOAD AREA CSV
    # --------------------------------------------------------

    @render.download(filename="UNet_Land_Cover_Area.csv")

    def download_area():

        current_area = area_data.get()
        if current_area is None:
            return
        return current_area.to_csv(index=False)


    # --------------------------------------------------------
    # DOWNLOAD ACCURACY CSV
    # --------------------------------------------------------

    @render.download(filename="UNet_Accuracy_Assessment.csv")

    def download_accuracy():

        current_accuracy = accuracy_data.get()
        if current_accuracy is None:
            return

        report = current_accuracy["classification_report"].reset_index().rename(columns={"index": "Class"})
        return report.to_csv(index=False)


# ============================================================
# 13. CREATE APPLICATION
# ============================================================

app = App(app_ui, server)
