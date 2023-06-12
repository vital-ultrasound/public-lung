#include "Widget_LUSclassificationp.h"
//#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QButtonGroup>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QCheckBox>
#include <QtInfoPanelTrafficLightBase.h>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QImage>
#include <QTimer>
#include "AspectRatioPixmapLabel.h"

Widget_LUSclassificationp::Widget_LUSclassificationp(
        QWidget *parent, Qt::WindowFlags f)
    : QtPluginWidgetBase(parent, f)
{

    mColorWithLevel = true;
    this->mWidgetLocation = WidgetLocation::top_right;
    mStreamTypes = ifind::InitialiseStreamTypeSetFromString("LUSclassificationp");
    mIsBuilt = false;

    mLabel = new QLabel("Text not set");
    mLabel->setStyleSheet(sQLabelStyle);


    mSummaryLabel = new QLabel("");
    mSummaryLabel->setStyleSheet(sQLabelStyle);

    this->hideCheckbox = new QCheckBox("Show Assistant",this);
    this->hideCheckbox->setChecked(true);
    this->hideCheckbox->setStyleSheet(sQCheckBoxStyle);

    // manual stuff
    {
        mPredictionsButtons = new QButtonGroup();
        mConfidencesButtons = new QButtonGroup();
    }

    mSubmitButton = new QPushButton("Submit");
    mSubmitButton->setStyleSheet(sQPushButtonStyle);
    mResetButton = new QPushButton("Reset");
    mResetButton->setStyleSheet(sQPushButtonStyle);

    QObject::connect(mResetButton, &QPushButton::released,
                     this, &Widget_LUSclassificationp::slot_reset);


    //--
    mLUSPicLabel = new AspectRatioPixmapLabel();


    QObject::connect(mResetButton, &QPushButton::released,
                     this->mLUSPicLabel, &AspectRatioPixmapLabel::slot_reset);

    QObject::connect(mLUSPicLabel, &AspectRatioPixmapLabel::signal_setZone,
                     this, &Widget_LUSclassificationp::slot_selectZone);

    //mLUSPicLabel->setScaledContents( true );
    //mLUSPicLabel->setSizePolicy( QSizePolicy::Ignored, QSizePolicy::Ignored );


    auto vLayout = new QVBoxLayout(this);
    vLayout->setContentsMargins(1, 1, 1, 1);
    vLayout->setSpacing(0);
    this->setLayout(vLayout);

    vLayout->addWidget(mLabel);
    this->AddInputStreamComboboxToLayout(vLayout);
    this->AddImageViewCheckboxToLayout(vLayout);
    //vLayout->addWidget(mSlider);
}

void Widget_LUSclassificationp::slot_reset(){
    infoPanelabel->setText("<please select a lung zone>");
    mPredictionsButtons->button(0)->setChecked(true); // check it to remove any other check
    mPredictionsButtons->setExclusive(false);
    mPredictionsButtons->button(0)->setChecked(false); // uncheck so that all are unchecked
    mPredictionsButtons->setExclusive(true);
    //
    mConfidencesButtons->button(0)->setChecked(true); // check it to remove any other check
    mConfidencesButtons->setExclusive(false);
    mConfidencesButtons->button(0)->setChecked(false); // uncheck so that all are unchecked
    mConfidencesButtons->setExclusive(true);
    //
    this->mSummaryLabel->setText("");
}

void Widget_LUSclassificationp::setArtPath(const QString &artPath)
{
    mArtPath = artPath;
    // fancy lung zones
    QString image_path = mArtPath + "/LUS_zones.png";
    QString label_image_path = mArtPath + "/LUS_zones_labels.png";

    QImage *image = new QImage(image_path);
    QImage *label_image = new QImage(label_image_path);

    mLUSPicLabel->setPixmap(QPixmap::fromImage(*image));
    mLUSPicLabel->setLabelPixmap(QPixmap::fromImage(*label_image));
    mLUSPicLabel->setBlank_image_filename(image_path);
}

void Widget_LUSclassificationp::slot_selectZone(int i, int s){
    Q_EMIT this->signal_zone_toggled(i, s);
    QString labeltext("  Selected: ");
    if (s == 1){
        labeltext += "R";
    } else if (s == 2){
        labeltext += "L";
    }
    labeltext += QString::number(i);
    infoPanelabel->setText(labeltext);
}

void Widget_LUSclassificationp::Build_AI_View(std::vector<std::string> &labelnames){

    QVBoxLayout * outmost_layout = reinterpret_cast<QVBoxLayout*>(this->layout());
    /// This will have bar graphs of the live scan plane values
    {
        QtInfoPanelTrafficLightBase::Configuration lungArtefactsTrafficLightConfig;

        lungArtefactsTrafficLightConfig.Mode =
                QtInfoPanelTrafficLightBase::Modes::ImmediateBarAbsolute;
        lungArtefactsTrafficLightConfig.LabelNames = labelnames;
        lungArtefactsTrafficLightConfig.NGridColumns = 2;

        lungArtefactsTrafficLightConfig.ValueColorsVector.push_back(
                    QtInfoPanelTrafficLightBase::ValueColors(
                        std::numeric_limits<double>::lowest(), // value
                        QColor("black"), // background colour
                        QColor("silver"))); // text colour

        lungArtefactsTrafficLightConfig.MetadataLabelsKey = "LUSClassificationP_labels";
        lungArtefactsTrafficLightConfig.MetadataValuesKey = "LUSClassificationP_confidences";
        lungArtefactsTrafficLightConfig.MetadataSplitCharacter = ',';

        auto infoPanel = new QtInfoPanelTrafficLightBase(lungArtefactsTrafficLightConfig, this);
        infoPanel->setColorWithLevel(this->colorWithLevel());
        infoPanel->SetStreamTypesFromStr("LUSClassificationP");

        // Add a box to enable / disable

        QVBoxLayout *AI_widget_layout = new QVBoxLayout();
        AI_widget_layout->addWidget(this->hideCheckbox);
        AI_widget_layout->addWidget(infoPanel);
        if (this->hideCheckbox->isChecked()){
            infoPanel->setVisible(true);
        } else {
            infoPanel->setVisible(false);
        }


        QObject::connect(this->hideCheckbox, &QCheckBox::toggled,
                         infoPanel, &QWidget::setVisible);

        outmost_layout->addLayout(AI_widget_layout);

        QObject::connect(this, &QtPluginWidgetBase::ImageAvailable,
                         infoPanel, &QtInfoPanelBase::SendImageToWidget);
    }

}

void Widget_LUSclassificationp::BuildUserSurveyButtons(std::vector<std::string> &labelnames){

    QVBoxLayout * outmost_layout = reinterpret_cast<QVBoxLayout*>(this->layout());
    /// Clinician selected regions
    // LUS zones
    {
        QVBoxLayout *l = new QVBoxLayout();
        l->addWidget(mLUSPicLabel, 1, Qt::AlignTop);
        infoPanelabel = new QLabel("<please select a lung zone>");
        infoPanelabel->setStyleSheet(sQLabelStyle);
        l->addWidget(this->infoPanelabel);
        outmost_layout->addLayout(l);
    }
    /*
    QGroupBox *regions = new QGroupBox("Zone");
    regions->setStyleSheet(sQGroupBoxStyle);
    QHBoxLayout *lungzones = new QHBoxLayout();
    // left first
    {
        QVBoxLayout *l = new QVBoxLayout();
        mLungZonesButtons->setExclusive(true);

        QStringList zone_names = {"L1", "L2", "L3", "L4", "L5", "L6"};
        for (int i=0; i < zone_names.size(); i++){
            QPushButton* r1 = new QPushButton(zone_names[i]);
            r1->setStyleSheet(sQPushButtonStyle);
            r1->setCheckable(true);
            mLungZonesButtons->addButton(r1);

            l->addWidget(r1, 1, Qt::AlignTop);
        }
        lungzones->addLayout(l);
    }


    // right first
    {
        QVBoxLayout *l = new QVBoxLayout();
        mLungZonesButtons->setExclusive(true);

        QStringList zone_names = {"R1", "R2", "R3", "R4", "R5", "R6"};
        for (int i=0; i < zone_names.size(); i++){
            QPushButton* r1 = new QPushButton(zone_names[i]);
            r1->setStyleSheet(sQPushButtonStyle);
            r1->setCheckable(true);
            mLungZonesButtons->addButton(r1);

            l->addWidget(r1, 1, Qt::AlignTop);
        }
        lungzones->addLayout(l);
    }
    regions->setLayout(lungzones);
    */

    /// Clinician prediction
    QGroupBox *user_prediction= new QGroupBox("User prediction");
    user_prediction->setStyleSheet(sQGroupBoxStyle);
    {
        QVBoxLayout *l = new QVBoxLayout();
        mPredictionsButtons->setExclusive(true);

        for (int i=0; i < labelnames.size(); i++){
            QPushButton* r1 = new QPushButton(QString::number(i) + " " + labelnames[i].c_str());
            r1->setStyleSheet(sQPushButtonStyle);
            r1->setCheckable(true);
            mPredictionsButtons->addButton(r1, i);
            QHBoxLayout *button_and_label = new QHBoxLayout();
            QLabel *legendlabel = new QLabel("█");
            QPalette palette = legendlabel->palette();
            palette.setColor(legendlabel->backgroundRole(), AspectRatioPixmapLabel::color_scores[i]);
            palette.setColor(legendlabel->foregroundRole(), AspectRatioPixmapLabel::color_scores[i]);
            legendlabel->setAutoFillBackground(true);
            legendlabel->setPalette(palette);
            button_and_label->addWidget(r1, 1);
            button_and_label->addWidget(legendlabel);
            l->addLayout(button_and_label, 1);
        }
        user_prediction->setLayout(l);
    }

    QHBoxLayout *sideways_l = new QHBoxLayout();
    //sideways_l->addWidget(regions);
    sideways_l->addWidget(user_prediction);
    outmost_layout->addLayout(sideways_l);

    /// Clinician level of confidence
    QGroupBox *user_confidence= new QGroupBox("User confidence");
    user_confidence->setStyleSheet(sQGroupBoxStyle);
    {
        QHBoxLayout *l = new QHBoxLayout();
        mConfidencesButtons->setExclusive(true);

        //QStringList c_names = {"Very confident", "Confident", "Not very confident", "Unsure"};
        QStringList c_names = {"+++", "+", "+-", "-"};
        for (int i=0; i < c_names.size(); i++){
            QPushButton* r1 = new QPushButton(c_names[i]);
            r1->setStyleSheet(sQPushButtonStyle);
            r1->setCheckable(true);
            mConfidencesButtons->addButton(r1, i);

            l->addWidget(r1, 1, Qt::AlignTop);
        }
        user_confidence->setLayout(l);
    }
    outmost_layout->addWidget(user_confidence);

    // A pushbutton to submit the result

    {
        QHBoxLayout *l = new QHBoxLayout();
        //QLabel *lb  = new QLabel("Time: 0s");
        //lb->setStyleSheet(sQLabelStyle);

        //l->addWidget(lb, 1, Qt::AlignTop);
        l->addWidget(mResetButton, 1, Qt::AlignTop);
        l->addWidget(mSubmitButton, 1, Qt::AlignTop);

        outmost_layout->addLayout(l);
    }

    {
        // summary
        QHBoxLayout *l = new QHBoxLayout();
        l->addWidget(mSummaryLabel, 1, Qt::AlignTop);
        outmost_layout->addLayout(l);
    }
}

void Widget_LUSclassificationp::showEvent(QShowEvent *event) {
    QWidget::showEvent( event );
    //mGv->fitInView(mScene->sceneRect(), Qt::KeepAspectRatio);
}

void Widget_LUSclassificationp::slot_showEvent(){
    this->showEvent(nullptr);
}

void Widget_LUSclassificationp::SendImageToWidgetImpl(ifind::Image::Pointer image){

    if (mIsBuilt == false){
        mIsBuilt = true;
        std::string labels = image->GetMetaData<std::string>((this->pluginName() +"_labels").toStdString().c_str());

        boost::char_separator<char> sep(",");
        boost::tokenizer< boost::char_separator<char> > tokens(labels, sep);
        std::vector<std::string> labelnames;
        BOOST_FOREACH (const std::string& t, tokens) {
            labelnames.push_back(t);
        }
        this->Build_AI_View(labelnames);
        this->BuildUserSurveyButtons(labelnames);

        /// This is required to make sure it resizes properly
        int timeout = 100;
        // send signal after 100 ms
        QTimer::singleShot(timeout, this, SLOT(slot_showEvent()));
    }


    std::stringstream stream;
    stream << "==" << this->mPluginName.toStdString() << "=="<<std::endl;
    stream << "Sending " << ifind::StreamTypeSetToString(this->mStreamTypes);

    mLabel->setText(stream.str().c_str());

    Q_EMIT this->ImageAvailable(image);
}

bool Widget_LUSclassificationp::colorWithLevel() const
{
    return mColorWithLevel;
}

void Widget_LUSclassificationp::setColorWithLevel(bool colorWithLevel)
{
    mColorWithLevel = colorWithLevel;
}
