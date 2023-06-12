#include "Plugin_LUSclassificationp.h"
#include <generated/plugin_lusclassificationp_config.h>
#include <ifindImagePeriodicTimer.h>
#include <QObject>
#include "Worker_LUSclassificationp.h"
#include <QLabel>
#include <QButtonGroup>
#include <QPushButton>
#include "AspectRatioPixmapLabel.h"
#include <QCheckBox>

Q_DECLARE_METATYPE(ifind::Image::Pointer)
Plugin_LUSclassificationp::Plugin_LUSclassificationp(QObject *parent) : Plugin(parent)
{
    {
        WorkerType::Pointer worker_ = WorkerType::New();
        worker_->params.python_folder = std::string(LUSclassificationP::getPythonFolder());
        this->worker = worker_;
    }
    this->mStreamTypes = ifind::InitialiseStreamTypeSetFromString("Input");
    this->setFrameRate(30); // by default 15fps
    this->Timer->SetDropFrames(true);
    this->mShowAssistantInitially = true;

    {
        // create widget
        WidgetType * mWidget_ = new WidgetType;
        mWidget_->setArtPath(QString(LUSclassificationP::getArtFolder()));
        mWidget_->setColorWithLevel(false);
        this->mWidget = mWidget_;

        /// connect the GUI to the worker
        WorkerType *w = std::dynamic_pointer_cast< WorkerType >(this->worker).get();

        //QObject::connect(mWidget_->mLungZonesButtons, QOverload<QAbstractButton *, bool>::of(&QButtonGroup::buttonToggled),
        //                 w, &WorkerType::slot_lungZoneButtonToggled);

        QObject::connect(mWidget_, &Widget_LUSclassificationp::signal_zone_toggled,
                         w, &WorkerType::slot_lungZoneButtonToggled);

        QObject::connect(mWidget_->mPredictionsButtons, QOverload<QAbstractButton *, bool>::of(&QButtonGroup::buttonToggled),
                         w, &WorkerType::slot_predictionsButtonToggled);

        QObject::connect(mWidget_->mConfidencesButtons, QOverload<QAbstractButton *, bool>::of(&QButtonGroup::buttonToggled),
                         w, &WorkerType::slot_confidencesButtonToggled);

        QObject::connect(mWidget_->mSubmitButton, &QPushButton::released,
                         w, &WorkerType::slot_Submit);

        QObject::connect(mWidget_->mResetButton, &QPushButton::released,
                         w, &WorkerType::slot_Reset);

        QObject::connect(w, &WorkerType::signal_submitData,
                         mWidget_->mLUSPicLabel, &AspectRatioPixmapLabel::slot_submitZoneScore);

        QObject::connect(w, &WorkerType::signal_updateSummary,
                         mWidget_->mSummaryLabel, &QLabel::setText);


    }
    {
        // create image widget
        ImageWidgetType * mWidget_ = new ImageWidgetType;
        this->mImageWidget = mWidget_;
        this->mImageWidget->SetStreamTypes(ifind::InitialiseStreamTypeSetFromString(this->GetCompactPluginName().toStdString()));
        this->mImageWidget->SetWidgetLocation(ImageWidgetType::WidgetLocation::visible); // by default, do not show

        // set image viewer default options:
        // overlays, colormaps, etc
        ImageWidgetType::Parameters default_params = mWidget_->Params();
        default_params.SetBaseLayer(0); // use the input image as background image
        default_params.SetOverlayLayer(-1); // show 1 layer on top of the background
        default_params.SetLutId(40);
        default_params.SetShowColorbar(false);
        mWidget_->SetParams(default_params);
    }

    this->SetDefaultArguments();
}

void Plugin_LUSclassificationp::Initialize(void){
    Plugin::Initialize();
    reinterpret_cast< ImageWidgetType *>(this->mImageWidget)->Initialize();
    this->worker->Initialize();
    WidgetType *w = reinterpret_cast< WidgetType * >(this->mWidget);
    w->hideCheckbox->setChecked(this->mShowAssistantInitially);

    //w->hideCheckbox->stateChanged(this->mShowAssistantInitially);

    // Retrieve the list of classes and create a blank image with them as meta data.
    ifind::Image::Pointer configuration = ifind::Image::New();
    configuration->SetMetaData<std::string>("PythonInitialized",this->GetPluginName().toStdString());
    Q_EMIT this->ConfigurationGenerated(configuration);

    this->Timer->Start(this->TimerInterval);
}

void Plugin_LUSclassificationp::slot_configurationReceived(ifind::Image::Pointer image)
{
    if (image->HasKey("PythonInitialized")){
        std::string whoInitialisedThePythonInterpreter =
                image->GetMetaData<std::string>("PythonInitialized");
        std::cout << "[WARNING from "<< this->GetPluginName().toStdString()
                  << "] Python interpreter already initialized by \""
                  << whoInitialisedThePythonInterpreter <<"\", no initialization required."<<std::endl;
        this->worker->setPythonInitialized(true);
    }

    if (image->HasKey("Python_gil_init")){
        std::cout << "[WARNING from "<< this->GetPluginName().toStdString() << "] Python Global Interpreter Lock already set by a previous plug-in."<<std::endl;
        this->worker->set_gil_init(1);
    }

    /// Pass on the message in case we need to "jump" over plug-ins
    Q_EMIT this->ConfigurationGenerated(image);
}
void Plugin_LUSclassificationp::SetDefaultArguments(){
    // arguments are defined with: name, placeholder for value, argument type,  description, default value


    mArguments.push_back({"modelname", "<*.h5>",
                          QString( Plugin::ArgumentType[3] ),
                          "Model file name (without folder).",
                          QString(std::dynamic_pointer_cast< WorkerType >(this->worker)->model.c_str())});

    mArguments.push_back({"nframes", "<val>",
                          QString( Plugin::ArgumentType[1] ),
                          "Number of frames in the buffer.",
                          QString::number(std::dynamic_pointer_cast< WorkerType >(this->worker)->buffer_capacity)});

    mArguments.push_back({"cropbounds", "xmin:ymin:width:height",
                          QString( Plugin::ArgumentType[3] ),
                          "set of four colon-delimited numbers with the pixels to define the crop bounds",
                          this->VectorToQString<double>(std::dynamic_pointer_cast< WorkerType >(this->worker)->cropBounds()).toStdString().c_str()});

    mArguments.push_back({"abscropbounds", "0/1",
                          QString( Plugin::ArgumentType[0] ),
                          "whether the crop bounds are provided in relative values (0 - in %) or absolute (1 -in pixels)",
                          QString::number(std::dynamic_pointer_cast< WorkerType >(this->worker)->absoluteCropBounds()).toStdString().c_str()});

    mArguments.push_back({"showassistant", "0/1",
                          QString( Plugin::ArgumentType[0] ),
                          "whether to show the AI assistant (1) or not (0)",
                          QString::number(this->mShowAssistantInitially).toStdString().c_str()});

    mArguments.push_back({"output", "<filename>",
                          QString( Plugin::ArgumentType[3] ),
                          "path to the output filename where results will be written",
                          std::dynamic_pointer_cast< WorkerType >(this->worker)->getOutput_filename().toStdString().c_str()});

}


void Plugin_LUSclassificationp::SetCommandLineArguments(int argc, char* argv[]){
    Plugin::SetCommandLineArguments(argc, argv);
    InputParser input(argc, argv, this->GetCompactPluginName().toLower().toStdString());

    {const std::string &argument = input.getCmdOption("modelname");
        if (!argument.empty()){
            std::dynamic_pointer_cast< WorkerType >(this->worker)->model = argument.c_str();
        }}
    {const std::string &argument = input.getCmdOption("nframes");
        if (!argument.empty()){
            std::dynamic_pointer_cast< WorkerType >(this->worker)->buffer_capacity = atoi(argument.c_str());
        }}

    {const std::string &argument = input.getCmdOption("cropbounds");
        if (!argument.empty()){
            std::dynamic_pointer_cast< WorkerType >(this->worker)->setCropBounds(this->QStringToVector<double>(argument.c_str()));
        }}

    {const std::string &argument = input.getCmdOption("showassistant");
        if (!argument.empty()){
            this->mShowAssistantInitially = atoi(argument.c_str());
        }}

    {const std::string &argument = input.getCmdOption("output");
        if (!argument.empty()){
            std::dynamic_pointer_cast< WorkerType >(this->worker)->setOutput_filename(argument.c_str());
        }}

    {const std::string &argument = input.getCmdOption("abscropbounds");
        if (!argument.empty()){
            std::dynamic_pointer_cast< WorkerType >(this->worker)->setAbsoluteCropBounds(atoi(argument.c_str()));
        }}

    // no need to add above since already in plugin
    {const std::string &argument = input.getCmdOption("verbose");
        if (!argument.empty()){
            std::dynamic_pointer_cast< WorkerType >(this->worker)->params.verbose= atoi(argument.c_str());
        }}

}

template <class T>
QString Plugin_LUSclassificationp::VectorToQString(std::vector<T> vec){
    QString out;

    for (T val : vec){
        out.push_back(QString::number(val) + ":");
    }
    // remove the last ':'
    int pos = out.lastIndexOf(QChar(':'));

    return out.left(pos);
}

template <class T>
std::vector<T> Plugin_LUSclassificationp::QStringToVector(QString str){
    std::vector<T> out;

    QStringList str_list = str.split(":");
    for (QString val : str_list){
        out.push_back(T(val.toDouble()));
    }

    return out;
}
extern "C"
{
#ifdef WIN32
/// Function to return an instance of a new LiveCompoundingFilter object
__declspec(dllexport) Plugin* construct()
{
    return new Plugin_LUSclassificationp();
}
#else
Plugin* construct()
{
    return new Plugin_LUSclassificationp();
}
#endif // WIN32
}


