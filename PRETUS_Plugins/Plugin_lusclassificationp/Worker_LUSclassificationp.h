#pragma once

#include <Worker.h>

#include <memory>
#include <mutex>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <chrono>
#include <boost/circular_buffer.hpp>
#include <QAbstractButton>


namespace py = pybind11;

class Worker_LUSclassificationp : public Worker{
    Q_OBJECT

public:

    struct InternalData {
        QString lungZone;
        QString user_prediction;
        QString user_confidence;
        QString AI_prediction;
        QString AI_confidence;
        int AI_prediction_i;
        int user_prediction_i;
        int user_time_ms;
        int ai_time_ms;

        InternalData(){
            lungZone ="R0";
            user_prediction ="N/A";
            user_confidence ="";
            AI_prediction ="N/A";
            AI_confidence ="";
            AI_prediction_i = -1;
            user_prediction_i = -1;
            user_time_ms = 0;
            ai_time_ms = 0;
        }

        QString toQString(){
            QString out = "Zone: " + lungZone +
                    "\nUser: " + user_prediction + "(" + user_confidence +")"+
                    "\nAI: " + AI_prediction + "(" + AI_confidence +")";
            return out;
        }
    };


    typedef Worker_LUSclassificationp            Self;
    typedef std::shared_ptr<Self>       Pointer;

    /** Constructor */
    static Pointer New(QObject *parent = 0) {
        return Pointer(new Worker_LUSclassificationp(parent));
    }

    struct Parameters : WorkerParameters
    {
        Parameters() {
            python_folder = "";
        }
        std::string python_folder;
    };
    ~Worker_LUSclassificationp();

    void Initialize();
    Parameters params;
    std::string model;
    unsigned long buffer_capacity; /// size of the image buffer

    std::vector<double> cropBounds() const;
    void setCropBounds(const std::vector<double> &cropBounds);
    std::vector<float> aratio() const;
    void setAratio(const std::vector<float> &aratio);
    std::vector<int> desiredSize() const;
    void setDesiredSize(const std::vector<int> &desiredSize);

    bool absoluteCropBounds() const;
    void setAbsoluteCropBounds(bool absoluteCropBounds);

    QString getOutput_filename() const;
    void setOutput_filename(const QString &value);

public Q_SLOTS:

    //virtual void slot_lungZoneButtonToggled(QAbstractButton *button, bool checked);
    virtual void slot_lungZoneButtonToggled(int i, int s);
    virtual void slot_predictionsButtonToggled(QAbstractButton *button, bool checked);
    virtual void slot_confidencesButtonToggled(QAbstractButton *button, bool checked);

    virtual void slot_Submit();
    virtual void slot_Reset();

Q_SIGNALS:

    void signal_submitData(int, int, int, double, int, double); // zone, side, user_score, user_confidence, ai_score, ai_confidence
    void signal_updateSummary(QString);

protected:
    Worker_LUSclassificationp(QObject* parent = 0);

    void doWork(ifind::Image::Pointer image);


    /**
     * @brief mCropBounds
     * x0. y0. width, height
     */
    std::vector<double> mCropBounds;
    std::vector<float> mAratio;
    std::vector<int> mDesiredSize;
    bool mAbsoluteCropBounds;
    QString output_filename;

private:

    std::mutex mutex_latestData;
    InternalData mLatestData;

    std::vector<InternalData> mAllZoneData;

    /// Python Functions
    py::object PyImageProcessingFunction;
    py::object PyPythonInitializeFunction;
    PyGILState_STATE gstate;
    QStringList labels;

    boost::circular_buffer<GrayImageType2D::Pointer> image_buffer;

    /* This buffer is a memory-contiguous block with all frames, to convert easily to numpy*/
    std::valarray<float> pixel_buffer;

    /**
     * @brief Update the values in pxel_buffer with the
     * current image buffer content
     */
    void UpdatePixelBuffer();

    template <class T>
    void array_to_vector(py::array &array, std::vector<T> &vector);

    QString makeSummaryString();


    // for time measurement
    std::chrono::steady_clock::time_point mTBegin;
    std::chrono::steady_clock::time_point mTEnd;




};


