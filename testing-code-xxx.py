# ======================
# UTILITY: CHECK UNKNOWN BRANDS
# ======================
def print_unknown_brand_products(processor: PDFProcessor):
    """Cetak produk dengan brand 'unknown'"""
    unknown_count = 0
    total_products = 0
    
    print("\n=== PRODUK DENGAN BRAND UNKNOWN ===")
    for product_name in processor.product_names:
        total_products += 1
        product_data = processor.get_product_data(product_name)
        if not product_data:
            continue
            
        brand = product_data.get('brand', 'unknown')
        if brand.lower() == 'unknown':
            unknown_count += 1
            print(f"- {product_name}")
    
    print(f"\nTotal produk: {total_products}")
    print(f"Produk unknown brand: {unknown_count}")
    print(f"Persentase unknown: {(unknown_count/total_products)*100:.2f}%")
    print("===================================")

# Contoh penggunaan:
if __name__ == "__main__":
    pdf_path = "dokumen/data1.pdf"  # Ganti dengan path PDF Anda
    processor = PDFProcessor()
    
    if processor.initialize_vector_store(pdf_path):
        print_unknown_brand_products(processor)