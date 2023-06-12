#include "Worker_LUSclassificationp.h"

#include <iostream>
#include <thread>

#include <QDebug>
#include <QThread>
#include <QDir>

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pylifecycle.h>
#include <itkImportImageFilter.h>
#include <chrono>
#include "pngutils.hxx"

Worker_LUSclassificationp::Worker_LUSclassificationp(QObject *parent) : Worker (parent){
    this->model = "models/model_001";
    this->buffer_capacity = 30;
    this->output_filename = "output.txt";

    mAbsoluteCropBounds = false; // by default relative
    mAratio = {488, 389}; // for Sonosite
    //mCropBounds = {96, 25, 488, 389}; // for sonosite videos: 640x480
    //mCropBounds = {0.15, 0.05, 0.75, 0.81}; // for sonosite videos: 640x480
    // mCropBounds = {525, 120, 1050, 840}; // for venugo 1920 × 1080 # Alberto
    // mCropBounds = {480, 120, 1130, 810}; // for venugo 1920 × 1080 # Nhat
    mCropBounds = {0.25, 0.1, 0.6, 0.75}; // for venugo 1920 × 1080
    mDesiredSize = {64, 64};

    this->mAllZoneData.resize(12); // this will store 1 to 6 right, then 1 to 6 left.
}

void Worker_LUSclassificationp::Initialize()
{
    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_LUSclassificationp::Initialize] initializing worker"<<std::endl;
    }

    if (!this->PythonInitialized)
    {
        try {
            py::initialize_interpreter();
        }
        catch (py::error_already_set const &python_err) {
            std::cout << python_err.what();
            return;
        }
        if (this->params.verbose){
            std::cout << "[VERBOSE][Worker_LUSclassificationp::Initialize] python interpreter initialized"<<std::endl;
        }
    }

    this->image_buffer.set_capacity(this->buffer_capacity);
    this->pixel_buffer.resize(0);

    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_LUSclassificationp::Initialize] Import python modules from "<<this->params.python_folder<<std::endl;
    }


    py::object getClassesFunction;

    try {
        py::exec("import sys");
        std::string command = "sys.path.append('" + this->params.python_folder + "')";
        py::exec(command.c_str());

        py::object processing = py::module::import("LUSclassificationp_worker");
        if (this->params.verbose){
            std::cout << "[VERBOSE][Worker_LUSclassificationp::Initialize] imported inference."<<std::endl;
        }
        this->PyImageProcessingFunction = processing.attr("dowork");
        this->PyPythonInitializeFunction = processing.attr("initialize");
        getClassesFunction = processing.attr("get_classes");

    }
    catch (std::exception const &python_err) {
        std::cout << "[ERROR][Worker_LUSclassificationp::Initialize] "<< python_err.what();
        return;
    }

    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_LUSclassificationp::Initialize] initialize LUS model"<<std::endl;
    }
    py::tuple sz = py::make_tuple(64, 64);
    this->PyPythonInitializeFunction(sz, this->params.python_folder + QString(QDir::separator()).toStdString() + this->model, bool(this->params.verbose));


    py::list pyclasses  = py::list(getClassesFunction());
    this->labels.clear();
    for (auto& el : pyclasses) this->labels.push_back(QString(el.cast<std::string>().data()));

    this->PythonInitialized = true;
}

Worker_LUSclassificationp::~Worker_LUSclassificationp(){
    py::finalize_interpreter();
}

void Worker_LUSclassificationp::doWork(ifind::Image::Pointer image){

    if (!this->PythonInitialized){
        return;
    }

    if (!Worker::gil_init) {
        this->set_gil_init(1);
        PyEval_SaveThread();

        ifind::Image::Pointer configuration = ifind::Image::New();
        configuration->SetMetaData<std::string>("Python_gil_init","True");
        Q_EMIT this->ConfigurationGenerated(configuration);
    }

    if (image == nullptr){
        if (this->params.verbose){
            std::cout << "Worker_LUSclassificationp::doWork() - input image was null" <<std::endl;
        }
        return;
    }

    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_LUSclassificationp::doWork()] process image"<<std::endl;
    }
    ifind::Image::Pointer image_ratio_adjusted;

    std::vector<int> absoluteCropBounds(4);
    if (this->absoluteCropBounds() == true){
        for (int i=0; i<mCropBounds.size(); i++) absoluteCropBounds[i] = mCropBounds[i];
    } else {
        // get the image size
        ifind::Image::SizeType imsize = image->GetLargestPossibleRegion().GetSize();
        absoluteCropBounds[0] = int(mCropBounds[0] * imsize[0]); // x0
        absoluteCropBounds[1] = int(mCropBounds[1] * imsize[1]); // y0
        absoluteCropBounds[2] = int(mCropBounds[2] * imsize[0]); // w
        absoluteCropBounds[3] = int(mCropBounds[3] * imsize[1]); // h

        if (this->params.verbose){
            std::cout << "\tWorker_RFSeg::doWork() computing absolute crop bounds"<<std::endl;

        }
    }
    if (this->params.verbose){
        ifind::Image::SizeType imsize = image->GetLargestPossibleRegion().GetSize();
        std::cout << "\tWorker_RFSeg::doWork() use bounds"<<std::endl;
        std::cout << "\t\timage size is "<< imsize[0] << "x" << imsize[1]<<std::endl;
        std::cout << "\t\tinput crop bounds are "<< mCropBounds[0] << ":" << mCropBounds[1]<< ":" << mCropBounds[2]<< ":" << mCropBounds[3]<<std::endl;
        std::cout << "\t\tabsolute crop bounds are "<< absoluteCropBounds[0] << ":" << absoluteCropBounds[1]<< ":" << absoluteCropBounds[2]<< ":" << absoluteCropBounds[3]<<std::endl;
    }

    /// Use the appropriate layer
    std::vector<std::string> layernames = image->GetLayerNames();
    int layer_idx = this->params.inputLayer;
    if (this->params.inputLayer <0){
        /// counting from the end
        layer_idx = image->GetNumberOfLayers() + this->params.inputLayer;
    }
    ifind::Image::Pointer layerImage = ifind::Image::New();
    layerImage->Graft(image->GetOverlay(layer_idx), layernames[layer_idx]);
    image_ratio_adjusted = this->CropImageToFixedAspectRatio(layerImage, &mAratio[0], &absoluteCropBounds[0]);

    //png::save_ifind_to_png_file<ifind::Image>(image_ratio_adjusted, "/home/ag09/data/VITAL/cpp_in_adjusted.png");
    // now resample to 64 64
    if (this->params.verbose){
        std::cout << "Worker_RFSeg::doWork() - resample"<<std::endl;
    }
    ifind::Image::Pointer image_ratio_adjusted_resampled  = this->ResampleToFixedSize(image_ratio_adjusted, &mDesiredSize[0]);
    //png::save_ifind_to_png_file<ifind::Image>(image_ratio_adjusted_resampled, "/home/ag09/data/VITAL/cpp_in_adjusted_resampled.png");
    this->params.out_spacing[0] = this->params.out_spacing[0] * (this->params.out_size[0] - 1 )/ (128 - 1);
    this->params.out_spacing[1] = this->params.out_spacing[1] * (this->params.out_size[1] - 1 )/ (128 - 1);
    this->params.out_size[0] = 128;
    this->params.out_size[1] = 128;
    /// extract central slice and crop

    ///----------------------------------------------------------------------------

    GrayImageType2D::Pointer image_2d = this->get2dimage(image_ratio_adjusted_resampled);

    //GrayImageType2D::Pointer image_2db = this->get2dimage(image);  /// Extract central slice
    std::vector <unsigned long> dims = {image_2d->GetLargestPossibleRegion().GetSize()[1],
                                        image_2d->GetLargestPossibleRegion().GetSize()[0]};
    if (!image_2d->GetBufferPointer() || (dims[0] < 5) || (dims[1] < 5))
    {
        qWarning() << "[VERBOSE][Worker_LUSclassificationp::doWork()] image buffer is invalid";
        return;
    }

    /// Here starts. Fill the buffer
    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_LUSclassificationp::doWork()] add image in the buffer"<<std::endl;
    }
    this->image_buffer.push_back(image_2d);

    if (this->image_buffer.full() == false){
        /// We can't do anything until the buffer is full
        if (this->params.verbose){
            std::cout << "[VERBOSE][Worker_LUSclassificationp::slot_Work] the image buffer only has "<< this->image_buffer.size() << " out of "<< this->buffer_capacity<< ", waiting to have more images"<<std::endl;
        }
        return;
    }
    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_LUSclassificationp::doWork()] the buffer is full, we can now analyse the sequence"<<std::endl;
    }

    /// The buffer is now full, we can create our 3D np array
    /// Input dimensions are swapped as ITK and numpy have inverted orders

    std::vector <unsigned long> dims_buffer = {this->buffer_capacity, image_2d->GetLargestPossibleRegion().GetSize()[1],
                                               image_2d->GetLargestPossibleRegion().GetSize()[0]};

    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_LUSclassificationp::doWork()] the buffer has a size of "<< dims_buffer[0]<<", "<< dims_buffer[1]<<", "<< dims_buffer[2]<<std::endl;
    }

    this->UpdatePixelBuffer();

    GrayImageType2D::Pointer attention_map;


    std::vector<float> detection_result;
    std::vector<float> localisation_result;

    /// This must be only around the python stuff! otherwise the c++ will not run on a separate thread
    this->gstate = PyGILState_Ensure();
    {
        //py::array numpyarray(dims_buffer, static_cast<GrayImageType2D::PixelType*>(&(this->pixel_buffer[0])));
        py::array numpyarray(dims_buffer, static_cast<float*>(&(this->pixel_buffer[0])));
        //py::tuple predictions = this->PyImageProcessingFunction(numpyarray);
        //py::array dr = py::array(predictions[0]);
        py::tuple out_tuple = this->PyImageProcessingFunction(numpyarray);
        py::array dr = out_tuple[0];
        py::array attention_array = out_tuple[1];
        //py::array dr = py::array(predictions[0]);
        this->array_to_vector<float>(dr, detection_result);

        //py::array lr = py::array(predictions[1]);
        //this->array_to_vector<float>(lr, localisation_result);

        typedef itk::ImportImageFilter< GrayImageType::PixelType, 2 >   ImportFilterType;
        ImportFilterType::SizeType imagesize;

        imagesize[0] = attention_array.shape(1);
        imagesize[1] = attention_array.shape(0);

        ImportFilterType::RegionType region;
        ImportFilterType::IndexType start;
        start.Fill(0);

        region.SetIndex(start);
        region.SetSize(imagesize);

        /// Define import filter
        ImportFilterType::Pointer importer = ImportFilterType::New();
        importer->SetOrigin( image_2d->GetOrigin() );
        importer->SetSpacing( image_2d->GetSpacing() );
        //std::cout <<"Spacing "<< image_2d->GetSpacing()[0] <<", "<< image_2d->GetSpacing()[1] << std::endl;
        importer->SetDirection( image_2d->GetDirection() );
        importer->SetRegion(region);

        GrayImageType::PixelType* localbuffer = static_cast<GrayImageType::PixelType*>(attention_array.mutable_data());
        /// Import the buffer
        importer->SetImportPointer(localbuffer, imagesize[0] * imagesize[1], false);
        importer->Update();

        /// Disconnect the output from the filter
        /// @todo Check if that is sufficient to release the numpy buffer, or if the buffer needs to obe memcpy'ed
        attention_map = importer->GetOutput();
        attention_map->DisconnectPipeline();
        attention_map->SetMetaDataDictionary(image_2d->GetMetaDataDictionary());

        if (this->params.verbose){
            std::cout << "[VERBOSE][Worker_LUSclassificationp::doWork()]  Atention added"<<std::endl;
            attention_map->Print(std::cout);
        }

        /// ---------- Get the contour of the ellipse ------------------------------------
        /// Create a 3D image with the 2D slice
        //png::save_ifind_to_png_file<GrayImageType2D>(attention_map, "/home/ag09/data/VITAL/attention_map.png");
        GrayImageType::Pointer attention_map_full = get3dimagefrom2d(attention_map);
        //png::save_ifind_to_png_file<GrayImageType>(attention_map_full, "/home/ag09/data/VITAL/attention_map_full.png");
        GrayImageType::Pointer attention_unresized= this->UndoResampleToFixedSize(attention_map_full, image, &absoluteCropBounds[0]);
        //png::save_ifind_to_png_file<GrayImageType>(attention_unresized, "/home/ag09/data/VITAL/attention_unresized.png");

        //exit(-1);

        if (this->params.verbose){
            std::cout << "[VERBOSE][Worker_LUSclassificationp::doWork()]  Unresize"<<std::endl;
            attention_map_full->Print(std::cout);
        }

        //GrayImageType::Pointer responsemap = this->UnAdjustImageSize(segmentation, image);
        //png::save_ifind_to_png_file<GrayImageType>(segmentation_unresized, "/home/ag09/data/VITAL/unresampled_seg.png");
        GrayImageType::Pointer attention = this->UndoCropImageToFixedAspectRatio(attention_unresized, image, &absoluteCropBounds[0]);

        if (this->params.verbose){
            std::cout << "[VERBOSE][Worker_LUSclassificationp::doWork()]  Uncrop"<<std::endl;
            attention->Print(std::cout);
        }

        image->GraftOverlay(attention.GetPointer(), image->GetNumberOfLayers(), "AttentionMap");
        image->SetMetaData<std::string>( mPluginName.toStdString() +"_output", QString::number(image->GetNumberOfLayers()).toStdString() );

    }

    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_LUSclassificationp::doWork()]  imported image"<<std::endl;
    }

    PyGILState_Release(this->gstate);

    /// Find max min and prediction
    double max_val = 0;
    int max_index = -1;
    std::vector<float>::const_iterator cit;
    for (cit = detection_result.begin(); cit != detection_result.end(); cit++){
        if (*cit > max_val){
            max_val = *cit;
            max_index = cit - detection_result.begin();
        }
    }
    double max_val_l = 0;
    int max_index_l = -1;
    int count = 0;
    if (max_index ==1){
        /// there is a b-line
        for (cit = localisation_result.begin(); cit != localisation_result.end(); cit++, count++){
            if (*cit > max_val_l){
                max_val_l = *cit;
                max_index_l = count;
            }
        }
    }

    // convert to a string
    QStringList confidences_str;
    for (unsigned int i=0; i<detection_result.size(); i++){
        double dr_normalised = detection_result[i]; ///max_val;
        confidences_str.append(QString::number(dr_normalised));
    }

    image->SetMetaData<std::string>( mPluginName.toStdString() +"_labels", this->labels.join(",").toStdString() );
    image->SetMetaData<std::string>( mPluginName.toStdString() + "_confidences", confidences_str.join(",").toStdString() );
    image->SetMetaData<std::string>( mPluginName.toStdString() +"_label", this->labels[max_index].toStdString() );
    image->SetMetaData<std::string>( mPluginName.toStdString() + "_confidence", confidences_str[max_index].toStdString() );

    mutex_latestData.lock();
    mLatestData.AI_prediction = this->labels[max_index];
    mLatestData.AI_prediction_i = max_index;
    mLatestData.AI_confidence = confidences_str[max_index];
    mutex_latestData.unlock();

    if (this->params.verbose){
        if (max_index == 1){
            std::cout << "[VERBOSE][Worker_LUSclassificationp::doWork()] Detection result = "<< this->labels[max_index].toStdString() << " ("<<max_val<<"), localised at frame "<< max_index_l<< std::endl;
        } else {
            std::cout << "[VERBOSE][Worker_LUSclassificationp::doWork()] Detection result = "<< this->labels[max_index].toStdString() << " ("<<max_val<<")"<< std::endl;
        }
    }

    Q_EMIT this->ImageProcessed(image);

}

QString Worker_LUSclassificationp::getOutput_filename() const
{
    return output_filename;
}

void Worker_LUSclassificationp::setOutput_filename(const QString &value)
{
    output_filename = value;
}

bool Worker_LUSclassificationp::absoluteCropBounds() const
{
    return mAbsoluteCropBounds;
}

void Worker_LUSclassificationp::setAbsoluteCropBounds(bool absoluteCropBounds)
{
    mAbsoluteCropBounds = absoluteCropBounds;
}

void Worker_LUSclassificationp::slot_lungZoneButtonToggled(int i, int s){
    //std::cout << "[Worker_LUSclassificationp::slot_predictionsButtonToggled] text is "<<button->text().toStdString()<< " with value "<< checked << std::endl;
    if (i > 0){
        QString lungZone("L");
        if (s==1){
            lungZone = "R";
        }
        lungZone += QString::number(i);
        mutex_latestData.lock();
        mLatestData.lungZone = lungZone ;
        mutex_latestData.unlock();

        // start timer until submit
        mTBegin = std::chrono::steady_clock::now();

    }
}

void Worker_LUSclassificationp::slot_predictionsButtonToggled(QAbstractButton *button, bool checked){
    //std::cout << "[Worker_LUSclassificationp::slot_predictionsButtonToggled] text is "<<button->text().toStdString()<< " with value "<< checked << std::endl;
    if (checked){
        mutex_latestData.lock();
        mLatestData.user_prediction = button->text();
        mLatestData.user_prediction_i= QString(button->text()[0]).toInt();
        mutex_latestData.unlock();
    }
}



void Worker_LUSclassificationp::slot_confidencesButtonToggled(QAbstractButton *button, bool checked){
    //std::cout << "[Worker_LUSclassificationp::slot_confidencesButtonToggled] text is "<<button->text().toStdString()<< " with value "<< checked << std::endl;
    if (checked){
        mutex_latestData.lock();
        mLatestData.user_confidence = button->text();
        mutex_latestData.unlock();
    }
}

void Worker_LUSclassificationp::slot_Reset(){
    // Reset the data
    mLatestData = InternalData();
    for (int i; i < this->mAllZoneData.size(); i++){
        this->mAllZoneData[i] = InternalData();
    }
}

void Worker_LUSclassificationp::slot_Submit(){
    int zone=-1, side=-1, score=-1;
    zone = QString(mLatestData.lungZone[1]).toInt();
    if (zone<=0){
        return;
    }

    if (mLatestData.lungZone[0]=="R"){
        side = 1;
    } else if (mLatestData.lungZone[0]=="L"){
        side = 2;
    }

    if (side<=0){
        return;
    }

    score = mLatestData.user_prediction_i; // or AI_prediction_i
    if (score<0){
        return;
    }

    if (mLatestData.user_confidence.length()==0){
        return;
    }

    double user_confidence = 1, ai_confidence = 0;
    int ai_score = -1;

    // compute timing
    mTEnd = std::chrono::steady_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(mTEnd - mTBegin).count();
    mutex_latestData.lock();
    mLatestData.user_time_ms = duration;
    mutex_latestData.unlock();

    Q_EMIT signal_submitData(zone, side, score, user_confidence, ai_score, ai_confidence);
    // now save all this data to disk, together with the video data
    // stored in the buffer

    // save to the 12 zone vector
    int id = (zone-1)+ (side-1)*6;
    this->mAllZoneData[id] = mLatestData;

    QString summary = this->makeSummaryString();
    Q_EMIT this->signal_updateSummary(summary);


}

QString Worker_LUSclassificationp::makeSummaryString(){
    int total_user_score = 0, total_ai_score = 0, total_user_time = 0;
    QString summary("Summary:\n");
    summary += "Zone\tUser\tAI\tU. t (s)\n";
    for (int i = 0; i < this->mAllZoneData.size(); i++){
        if (this->mAllZoneData[i].user_prediction_i >= 0){
            summary += this->mAllZoneData[i].lungZone +"\t"+
                    QString::number(this->mAllZoneData[i].user_prediction_i) +"\t"+
                    QString::number(this->mAllZoneData[i].AI_prediction_i) + "\t" + QString::number(this->mAllZoneData[i].user_time_ms/1000.0, 'f', 2);
            summary += "\n";
            total_user_score += this->mAllZoneData[i].user_prediction_i;
            total_ai_score += this->mAllZoneData[i].AI_prediction_i;
            total_user_time += this->mAllZoneData[i].user_time_ms;
        }
    }
    summary += "\nTotal:\n";
    summary += "\t" + QString::number(total_user_score) +"\t" + QString::number(total_ai_score) +"\t" + QString::number(total_user_time/1000.0, 'f', 2) + "\n";


    // now write to
    std::ofstream out;
    out.open(this->output_filename.toStdString());
    out << "Zone\tu.pred\tu.conf\tai.pred\tai.conf\tU. t (s)\n";
    for (unsigned int i = 0; i < this->mAllZoneData.size(); i++){
        if (this->mAllZoneData[i].user_prediction_i >= 0){
            out << this->mAllZoneData[i].lungZone.toStdString() << "\t"
            << this->mAllZoneData[i].user_prediction_i << "\t"
            << this->mAllZoneData[i].user_confidence.toStdString() << "\t"
            << this->mAllZoneData[i].AI_prediction_i << "\t"
            << this->mAllZoneData[i].AI_confidence.toDouble() << "\t"
            << this->mAllZoneData[i].user_time_ms <<std::endl;
        }
    }
    out.close();

    return summary;
}

template <class T>
void Worker_LUSclassificationp::array_to_vector(py::array &array, std::vector<T> &vector){
    vector.resize(array.size());
    T *ptr = (T *) array.data();
    for (int i=0 ;i<array.size(); i++){
        vector[i] =  *ptr;
        ptr++;
    }
}


void Worker_LUSclassificationp::UpdatePixelBuffer(){

    auto imsize = this->image_buffer[0]->GetLargestPossibleRegion().GetSize();
    int n_im_pixels = imsize[0]*imsize[1];
    int total_size = n_im_pixels*this->image_buffer.size();

    if (this->pixel_buffer.size() != total_size ){
        this->pixel_buffer.resize(total_size);
    }

    boost::circular_buffer<GrayImageType2D::Pointer>::const_iterator cit;
    int count = 0;
    for (cit = this->image_buffer.begin(); cit != this->image_buffer.end(); cit++, count++){
        int offset = count*n_im_pixels;
        GrayImageType2D::PixelType* current_pixel_pointer = static_cast<GrayImageType2D::PixelType*>((*cit)->GetBufferPointer());
        std::copy( current_pixel_pointer, current_pixel_pointer+n_im_pixels, std::begin(this->pixel_buffer) + offset );
    }

}


std::vector<int> Worker_LUSclassificationp::desiredSize() const
{
    return mDesiredSize;
}

void Worker_LUSclassificationp::setDesiredSize(const std::vector<int> &desiredSize)
{
    mDesiredSize = desiredSize;
}

std::vector<float> Worker_LUSclassificationp::aratio() const
{
    return mAratio;
}

void Worker_LUSclassificationp::setAratio(const std::vector<float> &aratio)
{
    mAratio = aratio;
}

std::vector<double> Worker_LUSclassificationp::cropBounds() const
{
    return mCropBounds;
}

void Worker_LUSclassificationp::setCropBounds(const std::vector<double> &cropBounds)
{
    mCropBounds = cropBounds;
}
