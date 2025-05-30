// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <vespa/eval/eval/value.h>
#include <vespa/searchcommon/attribute/iattributecontext.h>
#include <vespa/searchlib/queryeval/create_blueprint_params.h>
#include <vespa/searchlib/queryeval/i_element_gap_inspector.h>
#include <vespa/searchlib/queryeval/irequestcontext.h>
#include <vespa/vespalib/util/doom.h>

namespace search::fef {
class IObjectStore;
class IQueryEnvironment;
}

namespace proton {

class RequestContext : public search::queryeval::IRequestContext,
                       public search::queryeval::IElementGapInspector,
                       public search::attribute::IAttributeExecutor
{
public:
    using IAttributeContext = search::attribute::IAttributeContext;
    using IAttributeFunctor = search::attribute::IAttributeFunctor;
    using Doom = vespalib::Doom;
    RequestContext(const Doom& softDoom,
                   vespalib::ThreadBundle & threadBundle,
                   IAttributeContext& attributeContext,
                   const search::fef::IQueryEnvironment& query_env,
                   search::fef::IObjectStore& shared_store,
                   const search::queryeval::CreateBlueprintParams& create_blueprint_params,
                   const MetaStoreReadGuardSP * metaStoreReadGuard);

    const Doom & getDoom() const override { return _doom; }
    vespalib::ThreadBundle & thread_bundle() const override { return _thread_bundle; }
    const search::attribute::IAttributeVector *getAttribute(std::string_view name) const override;

    void asyncForAttribute(std::string_view name, std::unique_ptr<IAttributeFunctor> func) const override;
    const search::attribute::IAttributeVector *getAttributeStableEnum(std::string_view name) const override;

    const vespalib::eval::Value* get_query_tensor(const std::string& tensor_name) const override;

    const search::queryeval::CreateBlueprintParams& get_create_blueprint_params() const override {
        return _create_blueprint_params;
    }
    const MetaStoreReadGuardSP * getMetaStoreReadGuard() const override {
        return _metaStoreReadGuard;
    }

    const search::queryeval::IElementGapInspector& get_element_gap_inspector() const noexcept override;

    search::fef::ElementGap get_element_gap(uint32_t field_id) const noexcept override;

private:
    const Doom                                    _doom;
    vespalib::ThreadBundle                      & _thread_bundle;
    IAttributeContext                           & _attributeContext;
    const search::fef::IQueryEnvironment        & _query_env;
    search::fef::IObjectStore                   & _shared_store;
    search::queryeval::CreateBlueprintParams      _create_blueprint_params;
    const MetaStoreReadGuardSP                  * _metaStoreReadGuard;
};

}
