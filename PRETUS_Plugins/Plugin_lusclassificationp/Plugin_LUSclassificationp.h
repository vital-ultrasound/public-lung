#pragma once

#include <Plugin.h>
#include "Widget_LUSclassificationp.h"
#include "Worker_LUSclassificationp.h"
#include <QtVTKVisualization.h>

class Plugin_LUSclassificationp : public Plugin {
    Q_OBJECT

public:
    typedef Worker_LUSclassificationp WorkerType;
    typedef Widget_LUSclassificationp WidgetType;
    typedef QtVTKVisualization ImageWidgetType;
    Plugin_LUSclassificationp(QObject* parent = 0);

    QString GetPluginName(void){ return "LUS Classification P";}
    QString GetPluginDescription(void) {return "Automatic Classification in 5 classes -pytorch version.";}
    void SetCommandLineArguments(int argc, char* argv[]);

    void Initialize(void);

protected:
    virtual void SetDefaultArguments();
    template <class T> QString VectorToQString(std::vector<T> vec);
    template <class T> std::vector<T> QStringToVector(QString str);

public Q_SLOTS:
    virtual void slot_configurationReceived(ifind::Image::Pointer image);

private:
    bool mShowAssistantInitially;
};
