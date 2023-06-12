#pragma once
#include <QWidget>
#include <ifindImage.h>
#include <QtPluginWidgetBase.h>

class QLabel;
class AspectRatioPixmapLabel;
class QButtonGroup;
class QPushButton;
class QGraphicsView;
class QGraphicsScene;
class QCheckBox;

class Widget_LUSclassificationp : public QtPluginWidgetBase
{
    Q_OBJECT

public:
    Widget_LUSclassificationp(QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());

    virtual void SendImageToWidgetImpl(ifind::Image::Pointer image);

    bool colorWithLevel() const;
    void setColorWithLevel(bool colorWithLevel);

    QButtonGroup* mPredictionsButtons;
    QButtonGroup* mConfidencesButtons;
    QPushButton* mSubmitButton;
    QPushButton* mResetButton;
    AspectRatioPixmapLabel *mLUSPicLabel;
    QCheckBox *hideCheckbox;
    QLabel * mSummaryLabel;


    void setArtPath(const QString &artPath);

Q_SIGNALS:
    void signal_zone_toggled(int i, int s); // zone, side

protected:
    virtual void showEvent(QShowEvent *event);
    //virtual void resizeEvent(QResizeEvent *event);

protected Q_SLOTS:
    virtual void slot_showEvent();
    virtual void slot_selectZone(int i, int s);
    virtual void slot_reset();

private:
    // raw pointer to new object which will be deleted by QT hierarchy
    QLabel *mLabel;
    QLabel *infoPanelabel;
    bool mIsBuilt;
    bool mColorWithLevel;
    QGraphicsView *mGv;
    QGraphicsScene *mScene;
    // for the images
    QString mArtPath;



    /**
     * @brief Build the widget
     */
    void Build_AI_View(std::vector<std::string> &labelnames);
    void BuildUserSurveyButtons(std::vector<std::string> &labelnames);

};
