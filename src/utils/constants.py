DOC_TYPES = [
    "cover_letter",
    "certificate_of_quality",
    "packaging_specification",
    "bse_tse_declaration",
    "material_description",
    "supplier_qualification",
    "chain_of_custody",
    "certificate_of_processing",
    "unknown",
]


DOC_TYPE_DESCRIPTIONS = {
    "cover_letter": "Formal letters about product information, storage, or general communication.",
    "certificate_of_quality": "Lot numbers, batch numbers, manufacture dates, expiration dates, quality test results.",
    "packaging_specification": "Packaging materials, part numbers, dimensions, components, configuration details.",
    "bse_tse_declaration": "Animal-origin material declarations and TSE/BSE compliance statements.",
    "material_description": "Materials of construction, physical properties, sterilization compatibility.",
    "supplier_qualification": "Supplier approvals, audits, ISO certifications, qualification status.",
    "chain_of_custody": "Traceability, assembly history, shipment flow, custody records.",
    "certificate_of_processing": "Processing or sterilization records, treatment confirmation, process completion details.",
    "unknown": "Use only when the query is broad, unclear, or not tied to a known type.",
}